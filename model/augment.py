from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import imthread
import cv2
import os
import imutils
from imutils import paths

def generate_faces(image, amount=10):
    #resizing face image
    image = imutils.resize(image, width=600)

    # create image data augmentation generator
    samples = expand_dims(image, 0)
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=[0.5,1.0],
        zoom_range=[0.8,1]
        )

    # prepare iterator
    it = datagen.flow(samples, batch_size=1)

    def batch_generator(bin):
        batch = it.next()
        image = batch[0].astype('uint8')
        return image

    gen_count = list(range(0, amount))
    multi_threading = imthread.multi_threading(batch_generator, max_threads=amount)
    augmented_faces = multi_threading.start(gen_count)

    return augmented_faces

def augment_faces(dataset_dir='dataset', amount=10):
    # grab the paths to the input images in our dataset
    dataset = 'dataset'
    imagePaths = list(paths.list_images(dataset))

    names = []
    #creating augmentation directory if not exists
    for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-2]
        names.append(name)
        if not os.path.exists(str(dataset+'/'+name+'/'+name)):
            os.makedirs(str(dataset+'/'+name+'/'+name))

    #augmenting training images
    names = list(set(names))
    for name in names:
        images = os.listdir(dataset+'/'+name)
        images.pop(images.index(name))

        for image in images:
            print(f'[INFO] Augmenting {name} - {image}')
            raw_image = cv2.imread(f'{dataset}/{name}/{image}')

            generated_faces = generate_faces(raw_image, amount=amount)

            for i, aug_image in enumerate(generated_faces):
                image = image.split('.')[0]
                print(f'>> {dataset}/{name}/{name}/{image}_{i+1}.jpg')
                cv2.imwrite(f'{dataset}/{name}/{name}/{image}_{i+1}.jpg', aug_image)