#!/usr/bin/env python3
"""Publish only after every planned asset passes independent GitHub digest checks.

Run on the experiment host; pass credentials through stdin, never a command-line
argument or file. This process can outlive the initiating SSH connection.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import requests


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit_receipts(api, headers, directory):
    """Record completion on main without depending on the initiating laptop."""
    files = ['publication.json', 'upload-receipt.json']

    def call(method, path, **kw):
        response = requests.request(method, api + path, headers=headers, timeout=60, **kw)
        response.raise_for_status()
        return response.json()

    for attempt in range(6):
        try:
            head = call('GET', '/git/ref/heads/main')['object']['sha']
            parent = call('GET', '/git/commits/' + head)
            tree = []
            for name in files:
                blob = call('POST', '/git/blobs', json={
                    'content': (directory / name).read_text(), 'encoding': 'utf-8'})
                tree.append(dict(path='results/v06/artifacts_20260928/' + name,
                                 mode='100644', type='blob', sha=blob['sha']))
            new_tree = call('POST', '/git/trees', json={
                'base_tree': parent['tree']['sha'], 'tree': tree})
            if new_tree['sha'] == parent['tree']['sha']:
                return head  # Already recorded; safe to resume this step.
            commit = call('POST', '/git/commits', json={
                'message': 'Record verified semantic artifact release publication',
                'tree': new_tree['sha'], 'parents': [head]})
            updated = call('PATCH', '/git/refs/heads/main', json={
                'sha': commit['sha'], 'force': False})
            assert updated['object']['sha'] == commit['sha']
            return commit['sha']
        except requests.RequestException as error:
            code = error.response.status_code if error.response is not None else None
            if attempt == 5 or (code is not None and code not in (409, 422, 429) and code < 500):
                raise
            # Re-read main on retry; never overwrite somebody else's newer work.
            time.sleep(5 * (attempt + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--release-id', type=int, required=True)
    a = p.parse_args()
    token = sys.stdin.readline().strip()
    assert token
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28'}
    repo = 'KaiFeng-Frank/rgb-semantic-livo-mapping'
    api = 'https://api.github.com/repos/' + repo
    release = api + '/releases/' + str(a.release_id)
    d = a.directory
    while True:
        path = d / 'upload-receipt.json'
        receipt = json.loads(path.read_text()) if path.exists() else {}
        if receipt.get('complete'):
            break
        pid_path = d / 'upload.pid'
        if pid_path.exists():
            cmd = Path('/proc') / pid_path.read_text().strip() / 'cmdline'
            if not cmd.exists() or not cmd.read_bytes():
                raise RuntimeError('Uploader stopped before completion; draft is preserved.')
        time.sleep(10)
    manifest = json.loads((d / 'artifact-manifest.json').read_text())
    expected = {x['name']: x for x in manifest['assets']}
    for name in ['artifact-manifest.json', 'SHA256SUMS', 'RESTORE.md']:
        path = d / name
        expected[name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    archive_check = json.loads((d / 'archive-check.json').read_text())
    assert archive_check['all_pass'] and len(archive_check['archives']) == 9
    response = requests.get(release + '/assets', headers=headers, params={'per_page': 100}, timeout=60)
    response.raise_for_status()
    actual = {x['name']: x for x in response.json()}
    assert len(receipt['assets']) == len(expected)
    for name, item in expected.items():
        asset = actual[name]
        assert asset['state'] == 'uploaded' and asset['size'] == item['bytes'], name
        assert asset.get('digest') == 'sha256:' + item['sha256'], name
    # Read back the manifest as bytes through the GitHub asset endpoint.
    response = requests.get(api + '/releases/assets/' + str(actual['artifact-manifest.json']['id']),
                            headers={**headers, 'Accept': 'application/octet-stream'}, timeout=60)
    response.raise_for_status()
    assert hashlib.sha256(response.content).hexdigest() == sha(d / 'artifact-manifest.json')
    for name in ['upload-receipt.json', 'archive-check.json']:
        path = d / name
        if name in actual:
            assert actual[name].get('digest') == 'sha256:' + sha(path)
            continue
        response = requests.post('https://uploads.github.com/repos/' + repo + '/releases/' + str(a.release_id) + '/assets',
                                 params={'name': name}, headers={**headers, 'Content-Type': 'application/json'},
                                 data=path.read_bytes(), timeout=60)
        response.raise_for_status()
        assert response.json().get('digest') == 'sha256:' + sha(path)
    response = requests.patch(release, headers=headers, json={'draft': False, 'make_latest': 'false'}, timeout=60)
    response.raise_for_status()
    published = response.json()
    assert published['draft'] is False and published['tag_name'] == 'v0.6-artifacts-20260928'
    # The final check is unauthenticated: the user-facing release must be public.
    public = requests.get(api + '/releases/tags/v0.6-artifacts-20260928', timeout=60)
    public.raise_for_status()
    public = public.json()
    assert public['id'] == a.release_id and public['draft'] is False
    assert len(public['assets']) == len(expected) + 2
    assert all(x['state'] == 'uploaded' for x in public['assets'])
    result = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  url=public['html_url'], tag=public['tag_name'], assets=len(public['assets']),
                  total_bytes=sum(x['size'] for x in public['assets']),
                  all_server_digests_verified=True, manifest_download_verified=True,
                  archive_members_verified=sum(x['verified_members'] for x in archive_check['archives']),
                  public_access_verified=True)
    (d / 'publication.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    commit = commit_receipts(api, headers, d)
    (d / 'receipt-commit.json').write_text(json.dumps(dict(commit=commit)) + '\n')
    print('RECEIPTS_COMMITTED', commit, flush=True)


if __name__ == '__main__':
    main()
