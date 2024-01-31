import json
import argparse
import funcy
import os
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=False)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):

    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)

    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('train', type=str, help='Where to store COCO training annotations')
parser.add_argument('test', type=str, help='Where to store COCO test annotations')
parser.add_argument('valid', type=str, help='Where to store COCO valid annotations')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

parser.add_argument('--multi-class', dest='multi_class', action='store_true',
                    help='Split a multi-class dataset while preserving class distributions in train and test sets')

args = parser.parse_args()

def main(args):

    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)


        if args.multi_class:

            annotation_categories = funcy.lmap(lambda a: int(a['category_id']), annotations)

            #bottle neck 1
            #remove classes that has only one sample, because it can't be split into the training and testing sets
            annotation_categories =  funcy.lremove(lambda i: annotation_categories.count(i) <=1  , annotation_categories)

            annotations =  funcy.lremove(lambda i: i['category_id'] not in annotation_categories  , annotations)


            X_train, y_train, X_test, y_test = iterative_train_test_split(np.array([annotations]).T,np.array([ annotation_categories]).T, test_size = 1-args.split)

            save_coco(args.train, info, licenses, filter_images(images, X_train.reshape(-1)), X_train.reshape(-1).tolist(), categories)
            save_coco(args.test, info, licenses,  filter_images(images, X_test.reshape(-1)), X_test.reshape(-1).tolist(), categories)

            print("Saved {} entries in {} and {} in {}".format(len(X_train), args.train, len(X_test), args.test))
            
        else:

            #X_train, X_test = train_test_split(images, train_size=args.split)
            tr = []
            ts = []
            va = []
            print('images = ', images)
            for i in os.listdir('/content/Project1-2/train/images'):
              tr.append(i)
            for i in os.listdir('/content/Project1-2/test/images'):
              ts.append(i)
            for i in os.listdir('/content/Project1-2/valid/images'):
              va.append(i)
            print(len(tr),len(ts),len(va))
            print(images[0]['file_name'])
            tr_id = []
            ts_id = [] 
            va_id = []
            for i in tr:
              for j in images:
                if i == j['file_name']:
                  tr_id.append(j['id'])
            print('tr_id = ',len(tr_id))
            for i in ts:
              for j in images:
                if i == j['file_name']:
                  ts_id.append(j['id'])
            print('ts_id = ',len(ts_id))
            for i in va:
              for j in images:
                if i == j['file_name']:
                  va_id.append(j['id'])
            print('va_id = ',len(va_id))
            anns_train = filter_annotations(annotations, images)
            anns_test=filter_annotations(annotations, images)
            print('anns_train = ', anns_train)
            im_tr = []
            im_ts = []
            im_va = []
            an_tr = []
            an_ts = []
            an_va = []
            for num,i in enumerate(images):
              for j in tr_id:
                if i['id'] == j:
                  im_tr.append(i)
                  an_tr.append(annotations[num])
            print(len(im_tr))
            print(len(an_tr))
            for num,i in enumerate(images):
              for j in va_id:
                if i['id'] == j:
                  im_va.append(i)
                  an_va.append(annotations[num])
            print(len(im_va))
            print(len(an_va))
            for num,i in enumerate(images):
              for j in ts_id:
                if i['id'] == j:
                  im_ts.append(i)
                  an_ts.append(annotations[num])
            print(len(im_ts))
            print(len(an_ts))
            
            save_coco(args.train, info, licenses, im_tr, an_tr, categories)
            save_coco(args.test, info, licenses, im_ts, an_ts, categories)
            save_coco(args.valid, info, licenses, im_va, an_va, categories)


            print("Saved {} entries in {} and {} in {}".format(len(anns_train), args.train, len(anns_test), args.test))
            

if __name__ == "__main__":
    main(args)
