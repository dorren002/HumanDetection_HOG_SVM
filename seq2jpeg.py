import os.path
import fnmatch
import shutil


def seq2jpg(file, savepath, pa):
    f = open(file, 'rb')
    string = f.read().decode('latin-1')
    splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"
    # split .seq file into segment with the image prefix
    strlist = string.split(splitstring)
    f.close()
    count = 0
      for img in strlist:
        filename = "set01_" + pa +  "_" +str(count) + '.jpg'
        filenamewithpath = os.path.join(savepath, filename)
        # abandon the first one, which is filled with .seq header
        if count > 0:
            i = open(filenamewithpath, 'wb+')
            i.write(splitstring.encode('latin-1'))
            i.write(img.encode('latin-1'))
            i.close()
        count += 1


if __name__ == "__main__":
    rootdir = "F:\\HOGSVM\\set01"
    # walk in the rootdir, take down the .seq filename and filepath
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # check .seq file with suffix
            if fnmatch.fnmatch(filename, '*.seq'):
                # take down the filename with path of .seq file
                thefilename = os.path.join(parent, filename)
                # create the image folder by combining .seq file path with .seq filename
                savepath = "F:\\HOGSVM\\data\\image\\"
                seq2jpg(thefilename, savepath, filename.split('.')[0])
