import xml.etree.ElementTree as ET
from spectramanipulator.spectrum import Spectrum
import numpy as np

# bizare stuff...
def parse_XML_Spreadsheet(xml_text):
    def float_try_parse(num):
        try:
            return float(num.replace(',', '.').strip())
        except ValueError:
            return None

    try:
        root = ET.fromstring(xml_text)
        ssNs = '{urn:schemas-microsoft-com:office:spreadsheet}'

        # Table element
        t_el = None

        for el in root.iter(ssNs + 'Table'):
            t_el = el

        col_count = int(t_el.get(ssNs + 'ExpandedColumnCount'))

        if col_count < 2:
            return

        errs = col_count * [False]
        names = col_count * ['']

        buffer = []

        for row in t_el.iter(ssNs + 'Row'):

            vals = col_count * [None]

            # check whether current row contains only number, all of the Data elements are Type="Number"
            contains_numbers = all(el.attrib[ssNs + 'Type'] == 'Number' for el in row.iter(ssNs + 'Data'))

            cells = row.iter(ssNs + 'Cell')
            cell_idx = 0

            for cell in cells:

                index_attr = None
                try:
                    index_attr = cell.attrib[ssNs + "Index"]
                except KeyError:
                    pass

                if index_attr is not None:
                    new_cell_idx = int(index_attr) - 1
                    if contains_numbers:
                        for i in range(cell_idx, new_cell_idx):
                            errs[i] = True
                    else:
                        pass

                    cell_idx = new_cell_idx

                query = [el for el in cell.iter(ssNs + 'Data')]

                if len(query) == 0:
                    continue

                el_data = query[0]

                if contains_numbers:
                    vals[cell_idx] = float_try_parse(el_data.text)
                    if vals[cell_idx] is None:
                        errs[cell_idx] = True
                else:
                    names[cell_idx] = str(el_data.text)

                cell_idx += 1

            if cell_idx == 0:  # no cells in current row
                continue
            if cell_idx != col_count:
                for i in range(cell_idx, col_count):
                    errs[i] = True

            if contains_numbers:
                buffer.append(vals)

        row_count = len(buffer)

        if row_count < 2:
            return

        first_occur_idx = -1
        for i in range(col_count):
            if not errs[i]:
                first_occur_idx = i
                break

        if first_occur_idx == -1:
            return

        spectra = []

        for i in range(first_occur_idx + 1, col_count):
            # skip the wrong column
            if errs[i]:
                continue

            sp_data = []
            # group_name = self.get_filename() if self.filepath is not None else ''

            for j in range(row_count):
                sp_data.append((buffer[j][first_occur_idx], buffer[j][i]))

            sp = Spectrum(np.asarray(sp_data, dtype=np.float64), name=names[i])

            spectra.append(sp)

        return spectra if len(spectra) < 2 else [spectra]
    except Exception:
        raise

# if __name__ == "__main__":
#
#
#
# xml_data = """<?xml version="1.0"?>
# <?mso-application progid="Excel.Sheet"?>
# <Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
#  xmlns:o="urn:schemas-microsoft-com:office:office"
#  xmlns:x="urn:schemas-microsoft-com:office:excel"
#  xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"
#  xmlns:html="http://www.w3.org/TR/REC-html40">
#  <Styles>
#   <Style ss:ID="Default" ss:Name="Normal">
#    <Alignment ss:Vertical="Bottom"/>
#    <Borders/>
#    <Font ss:FontName="Calibri" x:CharSet="238" x:Family="Swiss" ss:Size="11"
#     ss:Color="#000000"/>
#    <Interior/>
#    <NumberFormat/>
#    <Protection/>
#   </Style>
#   <Style ss:ID="s66">
#    <NumberFormat ss:Format="0.0"/>
#   </Style>
#  </Styles>
#  <Worksheet ss:Name="Sheet1">
#   <Table ss:ExpandedColumnCount="4" ss:ExpandedRowCount="4"
#    ss:DefaultRowHeight="14.4">
#    <Row>
#     <Cell><Data ss:Type="String">wl</Data></Cell>
#     <Cell><Data ss:Type="String">a</Data></Cell>
#     <Cell><Data ss:Type="String">b</Data></Cell>
#     <Cell><Data ss:Type="String">c</Data></Cell>
#    </Row>
#    <Row>
#     <Cell ss:StyleID="s66"><Data ss:Type="Number">1.48613546846513</Data></Cell>
#     <Cell><Data ss:Type="Number">1567</Data></Cell>
#     <Cell><Data ss:Type="Number">11</Data></Cell>
#     <Cell><Data ss:Type="Number">266</Data></Cell>
#    </Row>
#    <Row>
#     <Cell><Data ss:Type="Number">5.87</Data></Cell>
#     <Cell><Data ss:Type="Number">458</Data></Cell>
#     <Cell><Data ss:Type="Number">48</Data></Cell>
#     <Cell><Data ss:Type="Number">55</Data></Cell>
#    </Row>
#    <Row>
#     <Cell><Data ss:Type="Number">67</Data></Cell>
#     <Cell><Data ss:Type="Number">4852</Data></Cell>
#     <Cell><Data ss:Type="Number">996</Data></Cell>
#     <Cell><Data ss:Type="Number">11.56</Data></Cell>
#    </Row>
#   </Table>
#  </Worksheet>
# </Workbook>"""
#
#
#
#
# # ee = ET.Element()
#
#
# def ns(element):
#     return '{urn:schemas-microsoft-com:office:spreadsheet}' + element
#
# def float_try_parse(num):
#     try:
#         return float(num.replace('.', '.').strip())
#     except ValueError:
#         return None
#
# root = ET.fromstring(xml_data)
# ssNs = '{urn:schemas-microsoft-com:office:spreadsheet}'
#
# # Table element
# t_el = None
#
# for el in root.iter(ssNs + 'Table'):
#     t_el = el
#
# col_count = int(t_el.get(ssNs + 'ExpandedColumnCount'))
#
# if col_count < 2:
#     print("return")
#     exit(0)
#
# errs = col_count * [False]
# names = col_count * ['']
#
# buffer = []
#
#
# for row in t_el.iter(ssNs + 'Row'):
#
#     vals = col_count * [None]
#
#     # check whether current row contains only number, all of the Data elements are Type="Number"
#     contains_numbers = all(el.attrib[ssNs + 'Type'] == 'Number' for el in row.iter(ssNs + 'Data'))
#
#     cells = row.iter(ssNs + 'Cell')
#     cell_idx = 0
#
#     for cell in cells:
#
#         index_attr = None
#         try:
#             index_attr = cell.attrib[ssNs + "Index"]
#         except KeyError:
#             pass
#
#         if index_attr is not None:
#             new_cell_idx = int(index_attr) - 1
#             if contains_numbers:
#                 for i in range(cell_idx, new_cell_idx):
#                     errs[i] = True
#             else:
#                 pass
#
#             cell_idx = new_cell_idx
#
#         query = [el for el in cell.iter(ssNs + 'Data')]
#
#         if len(query) == 0:
#             continue
#
#         el_data = query[0]
#
#         if contains_numbers:
#             vals[cell_idx] = float_try_parse(el_data.text)
#             if vals[cell_idx] is None:
#                 errs[cell_idx] = True
#         else:
#             names[cell_idx] = str(el_data.text)
#
#         cell_idx += 1
#
#     if cell_idx == 0:  # no cells in current row
#         continue
#     if cell_idx != col_count:
#         for i in range(cell_idx, col_count):
#             errs[i] = True
#
#     if contains_numbers:
#         buffer.append(vals)
#
# row_count = len(buffer)
#
# if row_count < 2:
#     print("return")
#     exit(0)
#
# first_occur_idx = -1
# for i in range(col_count):
#     if not errs[i]:
#         first_occur_idx = i
#         break
#
# if first_occur_idx == -1:
#     print("return")
#     exit(0)
#
# spectra = []
#
# for i in range(first_occur_idx + 1, col_count):
#     # skip the wrong column
#     if errs[i]:
#         continue
#
#     sp_data = []
#     # group_name = self.get_filename() if self.filepath is not None else ''
#
#     for j in range(row_count):
#         sp_data.append((buffer[j][first_occur_idx], buffer[j][i]))
#
#     sp = Spectrum(np.asarray(sp_data, dtype=np.float64), name=names[i])
#
#     spectra.append(sp)
#
#
#
#
#
#
