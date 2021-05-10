
import sys
# import os
from spectramanipulator.settings import Settings

ROOT_PATH = r"Software\Classes"
REG_PATH_EXT = f"{ROOT_PATH}\\{Settings.PROJECT_EXTENSION}"  # \.smpj
HKEY_CURRENT_USER_text = "HKEY_CURRENT_USER"

def associate_project_file():
    if sys.platform != 'win32':
        print(f"Association of {Settings.PROJECT_EXTENSION} project files files only implemented for Windows systems.")
        return

    import winreg as reg

    # help from https://stackoverflow.com/questions/1387769/create-registry-entry-to-associate-file-extension-with-application-in-c

    # [HKEY_CURRENT_USER\Software\Classes\blergcorp.blergapp.v1\shell\open\command]
    # @= "c:\path\to\app.exe \"%1\""
    # [HKEY_CURRENT_USER\Software\Classes\.blerg]
    # @= "blergcorp.blergapp.v1"

    try:
        REG_PATH_PROGRAM = f"{ROOT_PATH}\\{Settings.REG_PROGRAM_NAME}\\shell\\open\\command"

        reg.CreateKey(reg.HKEY_CURRENT_USER, REG_PATH_EXT)
        registry_key_ext = reg.OpenKey(reg.HKEY_CURRENT_USER, REG_PATH_EXT, 0, reg.KEY_WRITE)
        reg.SetValue(registry_key_ext, '', reg.REG_SZ, Settings.REG_PROGRAM_NAME)
        reg.CloseKey(registry_key_ext)

        print(f"Added {HKEY_CURRENT_USER_text}\\{REG_PATH_EXT}")

        # curr_dirr = os.path.dirname(os.path.realpath(__file__))
        # prog_path = os.path.join(curr_dirr, "spectramanipulator.pyw")
        prog_path = "spectramanipulator"

        py_exec = sys.executable  # find python executable
        # if py_exec.endswith('python.exe'):
        #     py_exec = os.path.join(os.path.dirname(sys.executable), 'pythonw.exe')  # use pythonw.exe

        # executable = f"\"{py_exec}\" \"{prog_path}\" \"%1\""  # path is "[pythonw.exe]" -m "[ssm.pyw]" "[path-to-file-arg]"
        executable = f"\"{py_exec}\" -m {prog_path} \"%1\""  # path is "[python.exe]" -m spectramanipulator "[path-to-file-arg]"
        # print(executable)

        reg.CreateKey(reg.HKEY_CURRENT_USER, REG_PATH_PROGRAM)
        registry_key_prog = reg.OpenKey(reg.HKEY_CURRENT_USER, REG_PATH_PROGRAM, 0, reg.KEY_WRITE)
        reg.SetValue(registry_key_prog, '', reg.REG_SZ, executable)
        reg.CloseKey(registry_key_prog)

        print(f"Added {HKEY_CURRENT_USER_text}\\{REG_PATH_PROGRAM}")

    except Exception as ex:
        print(ex.__str__())


def _delete_subkey(key0: int, key1: str, key2=""):
    # from https://stackoverflow.com/questions/38205784/python-how-to-delete-registry-key-and-subkeys-from-hklm-getting-error-5
    import winreg as reg

    if key2 == "":
        currentkey = key1
    else:
        currentkey = f"{key1}\\{key2}"

    open_key = reg.OpenKey(key0, currentkey, 0, reg.KEY_ALL_ACCESS)
    infokey = reg.QueryInfoKey(open_key)
    for x in range(0, infokey[0]):
        #  NOTE:: This code is to delete the key and all subkeys.
        #  If you just want to walk through them, then
        #  you should pass x to EnumKey. subkey = _winreg.EnumKey(open_key, x)
        #  Deleting the subkey will change the SubKey count used by EnumKey.
        #  We must always pass 0 to EnumKey so we
        #  always get back the new first SubKey.
        subkey = reg.EnumKey(open_key, 0)
        try:
            reg.DeleteKey(open_key, subkey)
            print(f"Removed {HKEY_CURRENT_USER_text}\\{currentkey}\\{subkey}")
        except:
            _delete_subkey(key0, currentkey, subkey)
            # no extra delete here since each call
            # to deleteSubkey will try to delete itself when its empty.

    reg.DeleteKey(open_key, "")
    open_key.Close()
    print(f"Removed {HKEY_CURRENT_USER_text}\\{currentkey}")


def remove_project_file_association():
    if sys.platform != 'win32':
        print(f"Association of {Settings.PROJECT_EXTENSION} project files files only implemented for Windows systems.")
        return

    import winreg as reg

    try:
        REG_PATH_PROGRAM = f"{ROOT_PATH}\\{Settings.REG_PROGRAM_NAME}"

        _delete_subkey(reg.HKEY_CURRENT_USER, REG_PATH_EXT)
        _delete_subkey(reg.HKEY_CURRENT_USER, REG_PATH_PROGRAM)

    except Exception as ex:
        print(ex.__str__())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--remove':
        remove_project_file_association()
        print(f"\nProject file {Settings.PROJECT_EXTENSION} association was removed from win registry.")
    else:
        associate_project_file()
        print(f"\nProject file {Settings.PROJECT_EXTENSION} association was added to win registry.")












