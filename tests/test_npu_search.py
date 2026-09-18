import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'Tools'/'npu-services'))
import npu_client
import search

def test_semantic_search_preserves_provenance(tmp_path,monkeypatch):
    monkeypatch.setattr(search,'NOTES_DIR',tmp_path)
    (tmp_path/'CHEM_1450.md').write_text('# Chemistry\n## Today\n- Diatomic molecules have two atoms.\n')
    def fake(action,body,timeout):
        assert action=='search'
        return {'results':body['rows']}
    monkeypatch.setattr(npu_client,'request',fake)
    hits=search.semantic_search('two atoms',include_transcripts=False)
    assert hits[0]['class_code']=='CHEM 1450'
    assert hits[0]['line']==3 and hits[0]['location']=='Today'
