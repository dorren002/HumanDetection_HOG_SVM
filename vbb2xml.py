import os, glob
import cv2
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from lxml import etree, objectify


def vbb_anno2dict(vbb_file, cam_id):
    # 通过os.path.basename获得路径的最后部分“文件名.扩展名”
    # 通过os.path.splitext获得文件名
    filename = os.path.splitext(os.path.basename(vbb_file))[0]

    # 定义字典对象annos
    annos = defaultdict(dict)
    vbb = loadmat(vbb_file)
    # object info in each frame: id, pos, occlusion, lock, posv
    objLists = vbb['A'][0][0][1][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]  # 可查看所有类别
    # person index
    person_index_list = np.where(np.array(objLbl) == "person")[0]  # 只选取类别为‘person’的xml
    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            frame_name = str(cam_id) + "_" + str(filename) + "_" + str(frame_id + 1) + ".jpg"
            annos[frame_name] = defaultdict(list)
            annos[frame_name]["id"] = frame_name
            annos[frame_name]["label"] = "person"
            for id, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                id = int(id[0][0]) - 1  # for matlab start from 1 not 0
                if not id in person_index_list:  # only use bbox whose label is person
                    continue
                pos = pos[0].tolist()
                occl = int(occl[0][0])
                annos[frame_name]["occlusion"].append(occl)
                annos[frame_name]["bbox"].append(pos)
            if not annos[frame_name]["bbox"]:
                del annos[frame_name]
    print(annos)
    return annos


def seq2img(annos, seq_file, outdir, cam_id):
    cap = cv2.VideoCapture(seq_file)
    index = 1
    # captured frame list
    v_id = os.path.splitext(os.path.basename(seq_file))[0]
    cap_frames_index = np.sort([int(os.path.splitext(id)[0].split("_")[2]) for id in annos.keys()])
    while True:
        ret, frame = cap.read()
        print(ret)
        if ret:
            if not index in cap_frames_index:
                index += 1
                continue
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outname = os.path.join(outdir, str(cam_id) + "_" + v_id + "_" + str(index) + ".jpg")
            print("Current frame: ", v_id, str(index))
            cv2.imwrite(outname, frame)
            height, width, _ = frame.shape
        else:
            break
        index += 1
    img_size = (width, height)
    return img_size


def instance2xml_base(anno, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/person'),
        E.filename(anno['id']),
        E.source(
            E.database('Caltech pedestrian'),
            E.annotation('Caltech pedestrian'),
            E.image('Caltech pedestrian'),
            E.url('None')
        ),
        E.size(
            E.width(640),
            E.height(480),
            E.depth(3)
        ),
        E.segmented(0),
    )
    for index, bbox in enumerate(anno['bbox']):
        bbox = [float(x) for x in bbox]
        if bbox_type == 'xyxy':
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
        else:
            xmin, ymin, xmax, ymax = bbox
        E = objectify.ElementMaker(annotate=False)
        anno_tree.append(
            E.object(
                E.name(anno['label']),
                E.bndbox(
                    E.xmin(xmin),
                    E.ymin(ymin),
                    E.xmax(xmax),
                    E.ymax(ymax)
                ),
                E.difficult(0),
                E.occlusion(anno["occlusion"][index])
            )
        )
    return anno_tree


def parse_anno_file(vbb_inputdir, vbb_outputdir):
    # annotation sub-directories in hda annotation input directory
    assert os.path.exists(vbb_inputdir)
    sub_dirs = os.listdir(vbb_inputdir)  # 对应set00,set01...
    for sub_dir in sub_dirs:
        cam_id = sub_dir  # set00 set01等
        # 获取某一个子set下面的所有vbb文件
        vbb_files = glob.glob(os.path.join(vbb_inputdir, sub_dir, "*.vbb"))
        for vbb_file in vbb_files:
            # 返回一个vbb文件中所有的帧的标注结果
            annos = vbb_anno2dict(vbb_file, cam_id)

            if annos:
                # 组成xml文件的存储文件夹，形如“/Users/chenguanghao/Desktop/Caltech/annotations/”
                vbb_outdir = vbb_outputdir

                # 如果不存在
                if not os.path.exists(vbb_outdir):
                    os.makedirs(vbb_outdir)

                for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                    if "bbox" in anno:
                        anno_tree = instance2xml_base(anno)
                        outfile = os.path.join(vbb_outdir, os.path.splitext(filename)[0] + ".xml")
                        print("Generating annotation xml file of picture: ", filename)
                        # 生成最终的xml文件，对应一张图片
                        etree.ElementTree(anno_tree).write(outfile, pretty_print=True)


def visualize_bbox(xml_file, img_file):
    import cv2
    tree = etree.parse(xml_file)
    # load image
    image = cv2.imread(img_file)
    origin = cv2.imread(img_file)
    # 获取一张图片的所有bbox
    for bbox in tree.xpath('//bndbox'):
        coord = []
        for corner in bbox.getchildren():
            coord.append(int(float(corner.text)))
        print(coord)
        cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
    # visualize image
    cv2.imshow("test", image)
    cv2.imshow('origin', origin)
    cv2.waitKey(0)


def main():
    vbb_inputdir = "F:/HOGSVM/annotations/"
    vbb_outputdir = "F:/HOGSVM/annotations/"
    parse_anno_file(vbb_inputdir, vbb_outputdir)

    """
    下面这段是测试代码
    """


    # """


if __name__ == "__main__":
    # main()
    xml_file = "F:/HOGSVM/annotations/set00_V000_526.xml"
    img_file = "F:/HOGSVM/set00/V000/526.jpg"
    visualize_bbox(xml_file, img_file)