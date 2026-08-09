"""Dashboard tests (BUILD_SPEC.md 11).

Runs against the DSP-only stack so the suite stays fast; the upload, manifest,
job and download paths under test are model-independent.
"""
from __future__ import annotations

import io
import os
import time
import zipfile

import pytest

os.environ.setdefault("VOICETONE_STACK", "vad,quality,noise")

from fastapi.testclient import TestClient  # noqa: E402

from app import auth  # noqa: E402
from app.main import app  # noqa: E402

AUTH = ("autoace", "changeme")


@pytest.fixture(scope="module")
def client():
    """A signed-in client. Logging in once sets the session cookie, which the
    TestClient then carries on every later request."""
    c = TestClient(app)
    r = c.post("/login", data={"username": AUTH[0], "password": AUTH[1], "next": "/"},
               follow_redirects=False)
    assert r.status_code == 303, r.text
    return c


@pytest.fixture
def anon():
    """A signed-out client.

    Function-scoped on purpose: the login tests below post credentials through
    this fixture, and a module-scoped client would keep the resulting cookie
    and quietly stop being anonymous for every later test.
    """
    return TestClient(app)


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in entries.items():
            zf.writestr(name, blob)
    return buf.getvalue()


def _wait(client, job_id: str, timeout: float = 300.0) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = client.get(f"/jobs/{job_id}/status").json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.3)
    pytest.fail("job did not finish in time")


# --------------------------------------------------------------------------
# auth and health
# --------------------------------------------------------------------------

def test_healthz_is_unauthenticated(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_signed_out_browser_is_redirected_to_login(anon):
    r = anon.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")


def test_signed_out_api_call_gets_json_not_html(anon):
    """A page script polling status must not have to parse a login page."""
    r = anon.get("/jobs/whatever/status")
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/json")


def test_login_page_renders(anon):
    r = anon.get("/login")
    assert r.status_code == 200
    assert "Sign in" in r.text


def test_wrong_password_is_rejected(anon):
    r = anon.post("/login", data={"username": "autoace", "password": "wrong"},
                  follow_redirects=False)
    assert r.status_code == 401
    assert "Incorrect username or password" in r.text
    assert auth.COOKIE_NAME not in r.cookies


def test_wrong_username_is_rejected(anon):
    r = anon.post("/login", data={"username": "nobody", "password": "changeme"},
                  follow_redirects=False)
    assert r.status_code == 401


def test_login_sets_a_session_cookie(anon):
    r = anon.post("/login", data={"username": "autoace", "password": "changeme"},
                  follow_redirects=False)
    assert r.status_code == 303
    assert auth.COOKIE_NAME in r.cookies


def test_logout_clears_the_session(client):
    c = TestClient(app)
    c.post("/login", data={"username": "autoace", "password": "changeme"})
    assert c.get("/").status_code == 200
    c.get("/logout")
    r = c.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 303


def test_tampered_cookie_is_rejected():
    """The signature is what stops a client editing its own session."""
    c = TestClient(app)
    c.cookies.set(auth.COOKIE_NAME, auth.issue("autoace")[:-4] + "AAAA")
    r = c.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 303


def test_expired_cookie_is_rejected(monkeypatch):
    token = auth.issue("autoace")
    assert auth.verify(token) == "autoace"
    # Capture the real clock BEFORE patching: calling time.time() inside the
    # replacement would re-enter the patch and recurse.
    later = time.time() + (auth.SESSION_HOURS + 1) * 3600
    monkeypatch.setattr(auth.time, "time", lambda: later)
    assert auth.verify(token) is None


def test_login_does_not_allow_an_open_redirect(anon):
    r = anon.post("/login", data={"username": "autoace", "password": "changeme",
                                  "next": "//evil.example.com/"},
                  follow_redirects=False)
    assert r.headers["location"] == "/"


def test_index_renders_when_authenticated(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Analyse" in r.text


def test_default_password_is_flagged_in_the_ui(client):
    """A deployment still on the shipped default is a finding, not a detail."""
    assert "Default credentials" in client.get("/").text


# --------------------------------------------------------------------------
# upload validation
# --------------------------------------------------------------------------

def test_upload_with_no_audio_is_rejected(client):
    blob = _zip({"readme.txt": b"nothing here"})
    r = client.post("/upload", files={"files": ("x.zip", blob, "application/zip")},
                    )
    assert r.status_code == 400
    assert "no supported audio" in r.json()["error"].lower()


def test_zip_bomb_is_refused(client):
    """A small archive that expands enormously must be refused before expansion."""
    blob = _zip({"bomb.wav": b"\0" * (300 * 1024 * 1024)})
    r = client.post("/upload", files={"files": ("bomb.zip", blob, "application/zip")},
                    )
    assert r.status_code == 413
    assert "expand" in r.json()["error"] or "compression" in r.json()["error"]


def test_path_traversal_in_archive_is_refused(client):
    blob = _zip({"../../escape.wav": b"RIFF"})
    r = client.post("/upload", files={"files": ("evil.zip", blob, "application/zip")},
                    )
    assert r.status_code in (400, 413)


# --------------------------------------------------------------------------
# the full job lifecycle
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def finished_job(client, samples):
    """A ZIP with nested folders, a manifest, and one unmatched row."""
    if len(samples) < 2:
        pytest.skip("samples not present")
    import csv
    import io as _io

    from voicetone.score import load_labels
    labels = load_labels(samples[0].parent / "labels.csv")

    rows = _io.StringIO()
    w = csv.writer(rows)
    w.writerow(["name", "result_json"])
    import json
    for name, val in labels.items():
        w.writerow([name, json.dumps(val)])
    w.writerow(["ghost.ogg", json.dumps(next(iter(labels.values())))])

    entries = {samples[0].name: samples[0].read_bytes(),
               f"nested/deeper/{samples[1].name}": samples[1].read_bytes(),
               "labels.csv": rows.getvalue().encode()}
    r = client.post("/upload",
                    files={"files": ("batch.zip", _zip(entries), "application/zip")},
                    )
    assert r.status_code == 200, r.text
    body = r.json()
    status = _wait(client, body["job"])
    return body, status


def test_nested_audio_is_found(finished_job):
    body, _ = finished_job
    assert body["total"] == 2, "audio in a nested folder must be discovered"


def test_manifest_is_checked_in_both_directions(finished_job):
    body, _ = finished_job
    assert "ghost.ogg" in body["unmatched_labels"]
    assert body["unlabelled_audio"] == [], "nested files should still match by name"


def test_job_completes_successfully(finished_job):
    _, status = finished_job
    assert status["status"] == "done"
    assert status["n_ok"] == 2
    assert status["n_failed"] == 0
    assert status["percent"] == 100


def test_results_page_renders(client, finished_job):
    body, _ = finished_job
    r = client.get(f"/jobs/{body['job']}")
    assert r.status_code == 200
    assert "Download CSV" in r.text
    assert "Scored against your manifest" in r.text, \
        "labels were supplied, so scoring must appear"


def test_json_download_carries_results_and_report(client, finished_job):
    body, _ = finished_job
    d = client.get(f"/jobs/{body['job']}/download.json").json()
    assert len(d["results"]) == 2
    assert "score_report" in d
    for row in d["results"]:
        assert row["result"] is not None
        assert "latents" in row and "stage_times" in row


def test_csv_download_has_one_column_per_field(client, finished_job):
    import csv as _csv
    body, _ = finished_job
    text = client.get(f"/jobs/{body['job']}/download.csv").content.decode("utf-8-sig")
    rows = list(_csv.reader(io.StringIO(text)))
    from voicetone.schema import FIELDS
    assert rows[0][:2] == ["name", "status"]
    for f in FIELDS:
        assert f in rows[0]
    assert len(rows) == 3


def test_original_filenames_are_preserved(client, finished_job, samples):
    body, _ = finished_job
    d = client.get(f"/jobs/{body['job']}/download.json").json()
    names = {r["name"] for r in d["results"]}
    assert names == {samples[0].name, samples[1].name}


def test_unknown_job_is_404(client):
    assert client.get("/jobs/deadbeef/status").status_code == 404
    assert client.get("/jobs/deadbeef/download.csv").status_code == 404


def test_retention_policy_matches_the_configured_window(client, finished_job):
    """Audio outlives the batch only when playback is enabled, and the UI says so.

    This replaces an earlier test that asserted immediate deletion. Playback was
    added deliberately, so the guarantee changed from "gone at once" to "gone
    within a bounded, stated window" -- and the window is enforced by
    test_expired_retention_purges_audio_from_disk below.
    """
    import tempfile
    from pathlib import Path

    from app.jobs import AUDIO_RETENTION_MIN, PLAYBACK_ENABLED, RETENTION
    body, _ = finished_job
    leftovers = [d for d in Path(tempfile.gettempdir()).glob("autoace_upload_*")
                 if any(d.rglob("*.ogg"))]
    if PLAYBACK_ENABLED:
        assert str(AUDIO_RETENTION_MIN) in RETENTION, \
            "the stated policy must name the actual window"
        assert "deleted" in RETENTION
    else:
        assert not leftovers, "with playback off, audio must not survive the batch"


# --------------------------------------------------------------------------
# archive hygiene and playback
# --------------------------------------------------------------------------

def test_macos_resource_forks_are_not_treated_as_audio(client, samples):
    """macOS Archive Utility shadows every entry with `__MACOSX/._name.ogg`.

    Those stubs carry the audio extension but contain an AppleDouble header, so
    an unfiltered batch of three calls arrives as six with three failures --
    which is exactly what happened before this was fixed.
    """
    if not samples:
        pytest.skip("samples not present")
    entries = {s.name: s.read_bytes() for s in samples[:2]}
    for s in samples[:2]:
        entries[f"__MACOSX/._{s.name}"] = b"\x00\x05\x16\x07Mac OS X resource fork"
    entries[".DS_Store"] = b"\x00\x00\x00\x01Bud1"

    r = client.post("/upload",
                    files={"files": ("Archive.zip", _zip(entries), "application/zip")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2, "resource forks must not be counted as audio"
    status = _wait(client, body["job"])
    assert status["n_failed"] == 0, "no junk file should reach the analyser"


def test_is_junk_covers_the_usual_archive_noise():
    from app.jobs import is_junk
    assert is_junk("__MACOSX/._call.ogg")
    assert is_junk("nested/__MACOSX/._call.ogg")
    assert is_junk("._call.ogg")
    assert is_junk(".DS_Store")
    assert is_junk("folder/Thumbs.db")
    assert not is_junk("call_001.ogg")
    assert not is_junk("nested/deeper/call_001.ogg")
    assert not is_junk("_call.ogg")          # single underscore is a real name


def test_audio_can_be_played_back(client, finished_job):
    """Playback is what lets an evaluator check a prediction by ear."""
    from app.jobs import PLAYBACK_ENABLED
    if not PLAYBACK_ENABLED:
        pytest.skip("audio retention disabled")
    body, _ = finished_job
    d = client.get(f"/jobs/{body['job']}/download.json").json()
    name = d["results"][0]["name"]
    r = client.get(f"/jobs/{body['job']}/audio", params={"name": name})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
    assert len(r.content) > 1000


def test_audio_route_cannot_be_walked_out_of(client, finished_job):
    """The name is a dict key, never a path fragment."""
    body, _ = finished_job
    for probe in ("../../etc/passwd", "/etc/passwd", "..%2f..%2fetc%2fpasswd",
                  "nope.ogg"):
        r = client.get(f"/jobs/{body['job']}/audio", params={"name": probe})
        assert r.status_code == 404, probe


def test_audio_requires_a_session(anon, finished_job):
    body, _ = finished_job
    r = anon.get(f"/jobs/{body['job']}/audio", params={"name": "call_001.ogg"})
    assert r.status_code == 401


def test_expired_retention_purges_audio_from_disk(client, samples):
    """The retention window is enforced, not just advertised."""
    from app.jobs import PLAYBACK_ENABLED
    from app.main import store
    if not PLAYBACK_ENABLED or not samples:
        pytest.skip("audio retention disabled or samples missing")

    r = client.post("/upload", files={"files": (samples[0].name,
                                                samples[0].read_bytes(),
                                                "audio/ogg")})
    job_id = r.json()["job"]
    _wait(client, job_id)
    job = store.get(job_id)
    root = job.root
    assert root is not None and root.exists(), "audio should be retained at first"

    job.audio_expires = 0.0            # pretend the window has closed
    store.sweep()
    assert not root.exists(), "expired audio must be deleted from disk"
    assert client.get(f"/jobs/{job_id}/audio",
                      params={"name": samples[0].name}).status_code in (404, 410)


def test_only_audio_extensions_are_analysed(client, samples):
    """The selection rule: extension against SUPPORTED, nothing else.

    An archive can contain anything -- documents, images, nested archives,
    editor backups, OS metadata -- and only the audio may reach the analyser.
    """
    if len(samples) < 2:
        pytest.skip("samples not present")
    entries = {s.name: s.read_bytes() for s in samples[:2]}
    entries.update({
        "docs/brief.pdf": b"%PDF-1.4 not audio",
        "docs/README.md": b"# notes",
        "sheet.xlsx": b"PK\x03\x04 spreadsheet",
        "logo.png": b"\x89PNG\r\n\x1a\n",
        "bundle.tar.gz": b"\x1f\x8b\x08 nested archive",
        "notes.txt": b"plain text",
        "__MACOSX/._" + samples[0].name: b"AppleDouble",
        ".DS_Store": b"Bud1",
    })
    r = client.post("/upload",
                    files={"files": ("mixed.zip", _zip(entries), "application/zip")})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["total"] == 2, "only the two audio files may be analysed"
    # Real user files that were ignored are reported; OS metadata is not,
    # because the user never put it there.
    assert set(body["skipped"]) == {"brief.pdf", "README.md", "sheet.xlsx",
                                    "logo.png", "bundle.tar.gz", "notes.txt"}

    status = _wait(client, body["job"])
    assert status["n_ok"] == 2 and status["n_failed"] == 0


def test_a_file_claiming_an_audio_extension_is_reported_not_hidden(client):
    """`.txt` renamed to `.wav` must fail loudly, not vanish.

    Selection is by extension precisely so this reaches the analyser: a corrupt
    recording and a mislabelled text file look identical until ffprobe opens
    them, and silently dropping the first would hide a real problem.
    """
    r = client.post("/upload", files={"files": (
        "mixed.zip", _zip({"call.wav": b"this is definitely not audio"},),
        "application/zip")})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["skipped"] == []
    status = _wait(client, body["job"])
    assert status["n_failed"] == 1, "it must surface as a failed row with a reason"


def test_a_running_batch_is_reachable_from_the_home_page(client, samples):
    """Processing continues in a background thread whether or not the browser
    is on the results page, so there has to be a way back to it. Without the
    recent-batches list a user who navigated away lost the job entirely."""
    if not samples:
        pytest.skip("samples not present")
    r = client.post("/upload", files={"files": (samples[0].name,
                                                samples[0].read_bytes(),
                                                "audio/ogg")})
    job_id = r.json()["job"]

    home = client.get("/").text
    assert "Recent uploads" in home
    assert f"/jobs/{job_id}" in home, "the batch must be linked from the home page"

    _wait(client, job_id)
    done = client.get("/").text
    assert f"/jobs/{job_id}" in done, "a finished batch must still be reachable"


def test_recent_batches_are_newest_first(client, samples):
    from app.main import store
    ids = [j.id for j in store.recent()]
    assert ids == sorted(ids, key=lambda i: -store.get(i).started)


def test_home_has_no_batch_list_before_anything_runs():
    """An empty state should not render an empty card."""
    from app.jobs import JobStore
    assert JobStore().recent() == []


def test_home_caps_the_batch_preview(client, samples):
    """An unbounded list on the home page pushes everything else off screen as
    usage builds up, so the home page previews and /history holds the rest."""
    from app.main import HOME_BATCHES
    if not samples:
        pytest.skip("samples not present")
    for _ in range(HOME_BATCHES + 2):
        client.post("/upload", files={"files": (samples[0].name,
                                                samples[0].read_bytes(),
                                                "audio/ogg")})
    home = client.get("/").text
    assert home.count('class="batch"') <= HOME_BATCHES
    assert "/history" in home, "there must be a way to see the rest"

    full = client.get("/history").text
    assert full.count('class="batch"') > HOME_BATCHES


def test_batches_page_has_an_empty_state():
    from app.main import app as fresh_app
    c = TestClient(fresh_app)
    c.post("/login", data={"username": "autoace", "password": "changeme"})
    r = c.get("/history")
    assert r.status_code == 200
    # Either populated by earlier tests, or showing the empty state -- never a
    # bare empty card.
    assert ('class="batch"' in r.text) or ("Nothing here yet" in r.text)


def test_batches_page_requires_a_session(anon):
    r = anon.get("/history", headers={"accept": "text/html"},
                 follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")


def test_selecting_a_file_clears_a_previous_error(client):
    """Regression: the "Choose at least one recording first" notice stayed on
    screen after a file was picked, so a valid selection sat next to an error
    telling the user they had not made one."""
    page = client.get("/").text
    assert "if (chosen.length) { err.style.display = 'none'; }" in page, \
        "render() must dismiss the error once files are chosen"


def test_a_single_recording_is_not_called_a_batch(client, samples):
    if not samples:
        pytest.skip("samples not present")
    r = client.post("/upload", files={"files": (samples[0].name,
                                                samples[0].read_bytes(),
                                                "audio/ogg")})
    job_id = r.json()["job"]
    _wait(client, job_id)
    page = client.get(f"/jobs/{job_id}").text
    assert "Batch results" not in page
    assert "Call results" in page


def test_the_job_id_is_not_shown_on_the_page(client, samples):
    """It is an internal handle. It belongs in the URL and the download
    filename, not in the reader's face."""
    if not samples:
        pytest.skip("samples not present")
    r = client.post("/upload", files={"files": (samples[0].name,
                                                samples[0].read_bytes(),
                                                "audio/ogg")})
    job_id = r.json()["job"]
    _wait(client, job_id)
    import re
    page = client.get(f"/jobs/{job_id}").text
    # Strip the places it legitimately belongs: download hrefs and the polling
    # constant. What is left is what a reader actually sees.
    visible = re.sub(r"<script.*?</script>", "", page, flags=re.S)
    visible = visible.replace(f"/jobs/{job_id}", "")
    assert job_id not in visible, "the raw job id should not be shown to the user"


def test_login_page_has_no_retention_paragraph(anon):
    assert "Sessions last 12 hours" not in anon.get("/login").text


def test_selected_files_can_be_removed_one_at_a_time(client):
    """A file picked by mistake has to be droppable without starting over.

    A browser's FileList is read-only and is replaced wholesale on every pick,
    so the page keeps its own array; these are the controls that drive it.
    """
    page = client.get("/").text
    assert 'class="pick-x"' in page, "each listed file needs its own remove button"
    assert "chosen.splice" in page, "the remove button must actually drop the file"
    assert 'id="clearAll"' in page, "a multi-file selection needs a clear-all"
    assert "input.value = ''" in page, \
        "the input must be emptied, or re-adding a removed file fires no change event"


def test_no_page_lectures_the_user_about_retention(client, anon):
    """Storage windows and process memory are operator concerns.

    They stay in the README and in jobs.py, off the pages someone is trying to
    upload calls on.
    """
    pages = {"/": client.get("/").text,
             "/history": client.get("/history").text,
             "/login": anon.get("/login").text}
    for where, text in pages.items():
        low = text.lower()
        for phrase in ("lost on restart", "in memory", "60 minutes", "deleted automatically"):
            assert phrase not in low, f"{where} still tells the user about {phrase!r}"


def test_the_footer_holds_up_on_a_short_page(client, anon):
    """Two regressions in one.

    The footer laid four siblings out around a flex:1 spacer, which wrapped
    into a ragged two-row block with a hole in it; and nothing pinned it to the
    bottom, so on a short page it stopped mid-screen with a bare rule floating
    over half a viewport of dead space.
    """
    page = client.get("/history").text
    assert "flex: 1 0 auto" in page, "main must grow so the footer reaches the bottom"
    assert '<span class="spacer"></span>' not in page.split("<footer>")[-1]
    assert "<footer>" not in anon.get("/login").text, \
        "the full-height sign-in screen suppresses the app footer"


def test_the_results_table_reads_in_plain_english(client, finished_job):
    """Column names are questions someone can answer, not schema abbreviations.

    "Conf.", "Overlap" and "Severity" name fields; they do not tell a reader
    what was measured.
    """
    body, _ = finished_job
    head = client.get(f"/jobs/{body['job']}").text.split("<thead>")[1].split("</thead>")[0]
    for phrase in ("How the customer sounded", "What the noise was",
                   "Talking over each other", "Confidence"):
        assert phrase in head, f"header {phrase!r} is missing"
    for gone in (">Conf.<", ">Type<", ">Severity<", ">Overlap<", ">Status<"):
        assert gone not in head, f"{gone} should have been replaced or removed"


def test_a_status_column_of_oks_is_not_a_column(client, finished_job):
    """A row that worked says so by having values in it. The one that did not
    is flagged on its own name cell, where the reader is already looking."""
    body, _ = finished_job
    page = client.get(f"/jobs/{body['job']}").text
    table = page.split("<table")[1].split("</table>")[0]
    assert 'class="state state-ok"' not in table, "a column of 'ok' carries nothing"
    assert 'class="tag' in table, "values should be tags, not bare words"
    assert "conf-track" in table, "confidence reads better as a meter than as a bare float"


def test_ordinal_columns_sort_by_rank_not_by_spelling(client, finished_job):
    """Alphabetically, High sorts above Low sorts above Medium.

    The ordinal columns therefore carry an explicit rank, taken from the same
    orderings the scorer uses for adjacent-class accuracy.
    """
    import re

    from voicetone.schema import TONE_ORDER
    body, _ = finished_job
    page = client.get(f"/jobs/{body['job']}").text
    table = page.split("<tbody>")[1].split("</tbody>")[0]
    cells = re.findall(r'<td data-sort="(\d+)">\s*<span class="tag [\w-]+">(\w+)</span>', table)
    # Yes/No cells carry a rank too, so pick out the ones holding a tone.
    tones = [(r, w) for r, w in cells if w.lower() in TONE_ORDER]
    assert tones, "tone cells must carry a sortable rank"
    for rank, word in tones:
        assert TONE_ORDER[int(rank)] == word.lower(), \
            f"{word} is ranked {rank}, which is {TONE_ORDER[int(rank)]}"


def test_the_scope_trace_tiles_without_a_seam(client):
    """The waveform loops by scrolling three copies of one path by one period.

    That only looks continuous if the envelope ends where it starts. Get it
    wrong and the trace visibly jumps once every cycle, which is the kind of
    thing that is obvious in motion and invisible in a diff.
    """
    import re
    page = client.get("/").text
    d = re.search(r'<path id="ioTrace" d="M([^"]+)"', page).group(1)
    pts = [tuple(float(v) for v in p.split(",")) for p in d.split(" L")]
    assert len(pts) == 68, "one period is 68 samples wide"

    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    assert 0 <= min(xs) and max(xs) < 404, "the period must fit the 404-unit screen"
    assert 0 < min(ys) and max(ys) < 214, "the trace must stay inside the screen"

    # Samples alternate sign about the centre line, so the first sample sits
    # above it and the last below. Equal distances mean a seamless handover.
    first, last = 107 - pts[0][1], pts[-1][1] - 107
    assert first == last, \
        f"envelope starts at {first} and ends at {last}; the loop will jump"


def test_an_upload_is_listed_by_what_was_uploaded(client, samples):
    """"3 recordings" is true of most rows and so identifies none of them.

    A ZIP is listed by its own name, not by its contents: "recordings.zip" is
    what the person who sent it will recognise.
    """
    if len(samples) < 2:
        pytest.skip("samples not present")
    blob = _zip({samples[0].name: samples[0].read_bytes(),
                 samples[1].name: samples[1].read_bytes()})
    r = client.post("/upload", files={"files": ("march-callbacks.zip", blob,
                                                "application/zip")})
    job_id = r.json()["job"]

    home = client.get("/").text
    assert 'class="batch-title" title="march-callbacks.zip"' in home, \
        "the list must name the archive that was uploaded"
    assert "march-callbacks.zip" in client.get("/history").text

    _wait(client, job_id)
    page = client.get(f"/jobs/{job_id}")
    assert "march-callbacks.zip" in page.text
    # The count survives too -- for an archive, name and size are different
    # facts and both are worth stating.
    assert "2 recordings" in page.text


def test_a_lone_recording_is_listed_by_its_filename(client, samples):
    if not samples:
        pytest.skip("samples not present")
    r = client.post("/upload", files={"files": (samples[0].name,
                                                samples[0].read_bytes(),
                                                "audio/ogg")})
    _wait(client, r.json()["job"])
    assert f'class="batch-title" title="{samples[0].name}"' in client.get("/").text


def test_job_titles_cover_the_shapes_an_upload_can_take(client):
    """The rule lives on Job, so it can be checked without running anything."""
    from app.jobs import Job
    assert Job(id="x", total=1, sources=["call_001.ogg"]).title == "call_001.ogg"
    assert Job(id="x", total=9, sources=["batch.zip"]).title == "batch.zip"
    assert Job(id="x", total=2, sources=["a.ogg", "b.ogg"]).title == "a.ogg and b.ogg"
    assert Job(id="x", total=4, sources=["a.ogg", "b.ogg", "c.ogg", "d.ogg"]).title \
        == "a.ogg and 3 others"
    # An older job carries no sources; it must still say something sensible.
    assert Job(id="x", total=3).title == "3 recordings"
    assert Job(id="x", total=1).title == "1 recording"
