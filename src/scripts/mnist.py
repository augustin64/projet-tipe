#!/usr/bin/python3
IMAGES_MAGIC_NUMBER = 2051
LABELS_MAGIC_NUMBER = 2049


def write_images_to_binary(images, filepath):
    byteorder = "big"

    bytes_ = IMAGES_MAGIC_NUMBER.to_bytes(4, byteorder=byteorder)
    bytes_ += (len(images)).to_bytes(4, byteorder=byteorder)
    bytes_ += len(images[0][0]).to_bytes(4, byteorder=byteorder)
    bytes_ += len(images[0]).to_bytes(4, byteorder=byteorder)

    for image in images:
        for row in image:
            for nb in row:
                bytes_ += int(nb).to_bytes(1, byteorder=byteorder)

    with open(filepath, "wb") as f:
        f.write(bytes_)


def write_labels_to_binary(labels, filepath):
    byteorder = "big"

    bytes_ = LABELS_MAGIC_NUMBER.to_bytes(4, byteorder=byteorder)
    bytes_ += (len(labels)).to_bytes(4, byteorder=byteorder)
    for label in labels:
        bytes_ += int(label).to_bytes(1, byteorder=byteorder)

    with open(filepath, "wb") as f:
        f.write(bytes_)
