import pandas as pd

# backright_path = 'C:\\Users\\18572\\Desktop\\backright.txt'
# frontright_path = 'C:\\Users\\18572\\Desktop\\frontright.txt'
# right_path = 'C:\\Users\\18572\\Desktop\\right.txt'
# br_path = 'C:\\Users\\18572\\Desktop\\br.txt'
# fr_path = 'C:\\Users\\18572\\Desktop\\fr.txt'
# r_path = 'C:\\Users\\18572\\Desktop\\r.txt'

# the path for the original coordinates
path1 = 'D:\\deeplabcut\\blender\\vae\\600edit_1.0\\model5\\result.txt'

# the path for the adjustments file
path2 = 'D:\\deeplabcut\\blender\\vae\\600edit_1.0\\model5\\data.txt'


def fourZeroFormat(n):
    if n >= 1000:
        return str(n)
    if n >= 100 and n < 1000:
        return "0" + str(n)
    if n >= 10 and n < 100:
        return "00" + str(n)
    if n < 10:
        return "000" + str(n)


# with h5py.File('C:\\Users\\18572\\Desktop\\Independent Study\\zebra\\test.h5', 'w') as hf:
#     hf.create_dataset("zebraGroundTruthLeftView",  data=all_frame_coords)

def create_hdf(pathOriginal, pathAdjustments, labelI, outputName):
    file = open(pathOriginal, "r")
    bulk_data = file.read()
    file.close()
    frame_lines = bulk_data.split('\n')

    file = open(pathAdjustments, "r")
    bulk_data = file.read()
    file.close()
    adjustment_lines = bulk_data.split('\n')

    all_frame_coords = []
    for i in range(0, len(frame_lines)):
        frame_line = frame_lines[i]
        frame_coords = frame_line.split("h")
        frame_coords = frame_coords[1:]
        row = []

        adjustment_line = adjustment_lines[i]
        adjustments = adjustment_line.split(" ")
        offsetX = int(adjustments[0])
        offsetY = int(adjustments[1])
        scaleFactor = float(adjustments[2])

        for coord in frame_coords:
            xy = coord.split(" ")
            row.append(int(xy[0]) * scaleFactor + offsetX)
            row.append(int(xy[1]) * scaleFactor + offsetY)

        all_frame_coords.append(row)

    image_list = []
    for row in range(len(all_frame_coords)):
        image_list.append("labeled-data\\" + str(labelI) + "\\" + fourZeroFormat(row) + ".png")

    annotated_order = ["nose", "neck",  "rshoulder", "relbow", "rfeet", "lshoulder", "lelbow", "lfeet",
                       "pelvis", "rpelvis",
                       "rknee", "rankle", "lpelvis", "lknee", "lankle", "lefteye", "righteye"]
    column = 0
    new = None
    for bodypart in annotated_order:
        a = []
        index = pd.MultiIndex.from_product(
            [["aclab"], [bodypart], ["x", "y"]],
            names=["scorer", "bodyparts", "coords"],
        )
        for i in range(len(all_frame_coords)):
            a.append(all_frame_coords[i][column:column + 2])
        frame = pd.DataFrame(a, columns=index, index=image_list)
        new = pd.concat([new, frame], axis=1)
        column += 2

    new.to_hdf(
        'D:\\deeplabcut\\blender\\vae\\600edit_1.0\\model5\\' + str(outputName) + '.h5',
        "df_with_missing",
        format="table",
        mode="w",
    )


# create_hdf(backright_path, 4, "backright")
# create_hdf(frontright_path, 5, "frontright")
# create_hdf(right_path, 6, "right")
# create_hdf(br_path, 7, "br")
# create_hdf(fr_path, 8, "fr")
# create_hdf(r_path, 9, "r")

create_hdf(path1, path2, "synmodel5", "CollectedData_aclab")