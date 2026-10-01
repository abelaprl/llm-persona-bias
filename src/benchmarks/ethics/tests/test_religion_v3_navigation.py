import unittest
import json
from unittest.mock import patch
import nbformat
from pathlib import Path
from test_religion_lm_v3_foundation import scratch_directory
from src.benchmarks.ethics.finalize_religion_v3_navigation import portable_links


class NavigationTests(unittest.TestCase):
    def test_rebuild_preserves_source_and_rejects_repeat_navigation_write(self):
        from src.benchmarks.ethics import rebuild_religion_v3_candidate as rebuild
        with scratch_directory() as root:
            report=root/'result/religion_lm_v3_candidate_fixture_overall'
            report.mkdir(parents=True)
            destination=root/'notebooks/religion/linguistic_marker_religion_ethics_v3_candidate_fixture_overall.ipynb'
            destination.parent.mkdir(parents=True)
            (report/'figures').mkdir()
            (report/'figures/x.png').write_bytes(b'fixture')
            notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('![x](figures\\x.png)')])
            source=report/'original.ipynb'
            nbformat.write(notebook,source)
            nbformat.write(notebook,destination)
            original=source.read_bytes()
            (report/'manifest.json').write_text(json.dumps(dict(default_notebook=source.name,notebook_path=str(destination))),encoding='utf-8')
            with patch.object(rebuild,'ROOT',root):
                rebuild.repair_notebook_links(report)
                self.assertEqual(source.read_bytes(),original)
                reviewed=nbformat.read(report/'religion_ethics_linguistic_markers_v3_reviewed_candidate.ipynb',as_version=4)
                self.assertEqual(reviewed.cells[0].source,'![x](figures/x.png)')
                actual=nbformat.read(destination,as_version=4)
                self.assertIn('../../result/religion_lm_v3_candidate_fixture_overall/figures/x.png',actual.cells[0].source)
                with self.assertRaises(FileExistsError):rebuild.repair_notebook_links(report)

    def test_windows_links_are_rebased_and_external_links_preserved(self):
        with scratch_directory() as root:
            source=root/'report';target=root/'notebooks';(source/'figures').mkdir(parents=True);target.mkdir()
            (source/'figures/x.png').write_bytes(b'fixture')
            text='![x](figures\\x.png) [site](https://example.com/a) [same](#anchor)'
            actual=portable_links(text,source,target)
            self.assertIn('](../report/figures/x.png)',actual)
            self.assertIn('](https://example.com/a)',actual)
            self.assertIn('](#anchor)',actual)

    def test_missing_link_is_not_silently_rewritten(self):
        with scratch_directory() as root:
            with self.assertRaises(FileNotFoundError):portable_links('[x](missing.png)',root,root)


if __name__=='__main__':unittest.main()
