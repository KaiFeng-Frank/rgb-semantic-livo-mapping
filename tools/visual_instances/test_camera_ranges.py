import io
import unittest
from unittest.mock import patch
import zipfile
from .fetch_camera_sweeps import RemoteZipFile


class FakeSession:
    def __init__(self,data,bad_status=False):self.data=data;self.requests=[];self.urls=[];self.bad_status=bad_status
    def get(self,url,headers,**kwargs):
        a,b=map(int,headers['Range'][6:].split('-'));self.requests.append((a,b));self.urls.append(url);data=self.data
        class Response:
            status_code=200 if self.bad_status else 206
            headers={'Content-Range':f'bytes {a}-{b}/{len(data)}'}
            raw=io.BytesIO(data[a:b+1])
            def __enter__(self):return self
            def __exit__(self,*args):pass
        response=Response();response.url='https://example.invalid/direct.zip';return response


class RangeTests(unittest.TestCase):
    def test_standard_zip_parser_and_crc_on_bounded_range_reader(self):
        f=io.BytesIO();payload=b'JPEG-test'*400000
        with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('CAM_FRONT/test.jpg',payload)
        session=FakeSession(f.getvalue())
        with patch('requests.Session',return_value=session):reader=RemoteZipFile('https://example.invalid/test.zip')
        with zipfile.ZipFile(reader) as z:self.assertEqual(z.read('CAM_FRONT/test.jpg'),payload)
        self.assertTrue(all(b-a+1<=1024*1024 for a,b in session.requests))
        self.assertLess(reader.downloaded,2*len(f.getvalue()))
        self.assertTrue(all(u=='https://example.invalid/direct.zip' for u in session.urls[1:]))

    def test_full_archive_response_and_unbounded_reads_are_rejected(self):
        with patch('requests.Session',return_value=FakeSession(b'01234567',True)),patch('time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'Range refused'):RemoteZipFile('https://example.invalid/test.zip')
        with patch('requests.Session',return_value=FakeSession(b'01234567')):reader=RemoteZipFile('https://example.invalid/test.zip')
        reader.size=24383899337
        with self.assertRaisesRegex(ValueError,'Unbounded'):reader.read()


if __name__=='__main__':unittest.main()
