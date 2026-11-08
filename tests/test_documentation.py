"""Comprueba que documentos/cuadernos consumen los artefactos del flujo vigente."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import unittest

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class DocumentationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('documentation_generator', ROOT / '10_generar_documentacion.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_public_forecast_schema_is_consumed_without_losing_territories(self):
        data = self.module.read_results()
        forecast = data['forecast']
        self.assertTrue({'province', 'forecast_quarter', 'prediction', 'lower_95', 'upper_95'}.issubset(forecast))
        final = forecast.loc[forecast.forecast_quarter.eq('2028Q4')]
        self.assertEqual(final.province_code.nunique(), 52)
        table = self.module.result_tables(data)['forecast']
        self.assertIn('Madrid', table)
        self.assertIn('Barcelona', table)
        self.assertIn('Superior 95', table)

    def test_current_notebooks_are_valid_and_contain_compilable_code(self):
        paths = sorted((ROOT / 'notebooks').glob('*.ipynb'))
        self.assertEqual({path.name for path in paths}, {
            '01_preparacion_y_calidad_datos.ipynb', '02_analisis_de_variables.ipynb',
            '03_seleccion_modelos_variables.ipynb', '04_evaluacion_y_calibracion.ipynb',
            '05_prevision_hasta_2028.ipynb',
        })
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code':
                    compile(cell.source, f'{path.name}:cell{index}', 'exec')

    def test_documentation_refers_to_actual_selection_and_test_command(self):
        selection = json.loads((ROOT / 'data/reports/unified/selection.json').read_text(encoding='utf-8'))
        protocol = (ROOT / 'docs/protocolo_experimental.md').read_text(encoding='utf-8')
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertIn(selection['selection_id'], protocol)
        self.assertIn('-m unittest discover -s tests -v', readme)
        self.assertNotIn('-m pytest', readme)


if __name__ == '__main__':
    unittest.main()
