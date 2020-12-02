from PyQt5 import uic
import glob
for fname in glob.glob("SSM/dialogs/*.ui", recursive=True):
    print("converting", fname)
    fin = open(fname, 'r')
    fout = open(fname.replace(".ui", ".py"), 'w')
    uic.compileUi(fin, fout, execute=True)
    fin.close()
    fout.close()


for fname in glob.glob("SSM/*.ui", recursive=True):
    print("converting", fname)
    fin = open(fname, 'r')
    fout = open(fname.replace(".ui", ".py"), 'w')
    uic.compileUi(fin, fout, execute=True)
    fin.close()
    fout.close()