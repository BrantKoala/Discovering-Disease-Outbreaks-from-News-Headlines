from IPython.display import Image

Image("database\\world Zika outbreak.png")

# import nbformat
# from nbconvert.preprocessors import ExecutePreprocessor
# from nbconvert import PDFExporter
#
# notebook_filename = "phase_5.ipynb"
#
# with open(notebook_filename) as f:
#     nb = nbformat.read(f, as_version=4)
#
# ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
#
# ep.preprocess(nb, {'metadata': {'path': 'database/'}})
#
# pdf_exporter = PDFExporter()
#
# pdf_data, resources = pdf_exporter.from_notebook_node(nb)
#
# with open("notebook.pdf", "wb") as f:
#     f.write(pdf_data)
