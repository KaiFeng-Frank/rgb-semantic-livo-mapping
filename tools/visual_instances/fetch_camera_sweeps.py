"""Fetch selected ZIP members with strict HTTP Range bounds; never fetch full ZIP.

Standard-library ZIP parsing/CRC verification, fixed public source revision, no
extraction paths from the archive. This tool is also runnable standalone remotely.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
import zipfile
import zlib
import requests

URL='https://hf-mirror.com/datasets/Xiaodong/Nuscenes-v1.0-trainval-CAM_FRONT_Sweeps/resolve/11f15b33699f8a5fc8ad7ed042a715d17df146d6/sweeps.zip'


class RemoteZipFile(io.RawIOBase):
    def __init__(self,url=URL):
        self.url=url;self.session=requests.Session();self.position=0;self.size=None;self.downloaded=0;self.cache=OrderedDict();self.block=1024*1024
        self._range(0,3)
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if target<0 or target>self.size:raise ValueError('Out-of-bounds ZIP seek')
        self.position=target;return target
    def _range(self,start,end):
        if self.downloaded+end-start+1>1000000000:raise ValueError('1GB network budget exhausted')
        for attempt in range(3):
            try:
                # Query key avoids caches serving a different Range at the same
                # resolve URL. The pinned archive content is unchanged.
                with self.session.get(self.url+f'?download=true&range_start={start}&range_end={end}',
                        headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(10,30)) as r:
                    if r.status_code!=206:raise ValueError(f'Range refused: HTTP {r.status_code}')
                    m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
                    if not m or (int(m[1]),int(m[2]))!=(start,end):raise ValueError('Incorrect Content-Range')
                    size=int(m[3])
                    if self.size is not None and size!=self.size:raise ValueError('Remote archive changed')
                    self.size=size;content=r.raw.read(end-start+2)
                    self.downloaded+=len(content)
                    if len(content)!=end-start+1:raise ValueError('Truncated/oversized range')
                    return content
            except (requests.RequestException,ValueError):
                if attempt==2:raise
                time.sleep(1+attempt)
    def read(self,n=-1):
        # zipfile asks for read-to-EOF while parsing the small end record.
        # Permit that only when the remaining suffix itself is bounded.
        if n<0:n=self.size-self.position
        if n>64000000:raise ValueError('Unbounded/oversized ZIP read refused')
        end=min(self.size,self.position+n);pieces=[]
        while self.position<end:
            k=self.position//self.block;start=k*self.block
            if k not in self.cache:
                self.cache[k]=self._range(start,min(self.size-1,start+self.block-1))
                if len(self.cache)>4:self.cache.popitem(last=False)
            self.cache.move_to_end(k);stop=min(end,start+len(self.cache[k]))
            pieces.append(self.cache[k][self.position-start:stop-start]);self.position=stop
        return b''.join(pieces)


def fetch(paths,out):
    requested=json.loads(paths.read_text())
    if len(requested)>2000 or len(set(requested))!=len(requested):raise ValueError('Invalid bounded request')
    wanted={Path(p).name for p in requested}
    if len(wanted)!=len(requested) or any(not p.startswith('sweeps/CAM_FRONT/') or not p.endswith('.jpg') for p in requested):raise ValueError('Unexpected camera path')
    out.mkdir(parents=True,exist_ok=True);images=out/'images';images.mkdir(exist_ok=True)
    remote=RemoteZipFile();results=[];stored=0
    with zipfile.ZipFile(remote) as z:
        members=[m for m in z.infolist() if Path(m.filename).name in wanted]
        if len(members)!=len(wanted) or {Path(m.filename).name for m in members}!=wanted:raise ValueError('Missing/duplicate archive members')
        if sum(m.file_size for m in members)>700000000:raise ValueError('700MB selected-image budget exceeded')
        for i,m in enumerate(sorted(members,key=lambda m:m.header_offset)):
            if m.file_size>2000000 or m.is_dir():raise ValueError('Invalid image member')
            if shutil.disk_usage(out).free<4000000000:raise ValueError('Less than 4GB free')
            p=images/Path(m.filename).name
            data=p.read_bytes() if p.exists() else z.read(m)
            if len(data)!=m.file_size or zlib.crc32(data)!=m.CRC or not data.startswith(b'\xff\xd8'):raise ValueError('JPEG/ZIP verification failed')
            if not p.exists():
                partial=p.with_suffix('.partial');partial.write_bytes(data);partial.replace(p)
            stored+=len(data);results.append(dict(name=p.name,size=len(data),crc32=m.CRC,sha256=hashlib.sha256(data).hexdigest()))
            if i%25==0:print(dict(done=i+1,total=len(members),stored_bytes=stored,network_bytes=remote.downloaded),flush=True)
    result=dict(source=URL,archive_size=remote.size,requested_sha256=hashlib.sha256(paths.read_bytes()).hexdigest(),
        files=results,stored_bytes=stored,network_bytes=remote.downloaded,complete=True)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(dict(done=len(results),stored_bytes=stored,network_bytes=remote.downloaded),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paths',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();fetch(a.paths,a.out)
