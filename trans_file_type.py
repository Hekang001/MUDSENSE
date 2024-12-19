import comtypes.client
import os
from changeOffice import Change


'''转换wps文件格式'''
def Uniform_filetype(path):
    file_ls = []
    for root, dirs, files in os.walk(path):
        root_file_ls = [os.path.join(root, file) for file in files]
        file_ls += root_file_ls

    for file in file_ls:
        if file.endswith(".wps"):
            wps = comtypes.client.CreateObject("KWPS.Application")
            doc = wps.Documents.Open(file)
            doc.SaveAs(file[:-4] + ".doc", FileFormat=0)  # 0 表示保存为DOC格式
            doc.Close()
            wps.Quit()
            os.remove(file)
        if file.endswith(".dps"):
            os.rename(file, file[:-4] + ".ppt")
    # doc到docx xls到xlsx ppt到pptx
    c = Change(path)
    c.doc2docx()
    c.xls2xlsx()
    c.ppt2pptx()
    print("转换完成！\n")


'''转换wps/dps/doc/ppt为pptx/docx'''
if __name__ == "__main__":
    rel_path="\data"
    abs_path = os.getcwd()+rel_path
    Uniform_filetype(abs_path)

