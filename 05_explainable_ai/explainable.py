from argparse import ArgumentParser
'''
Reading input data (training, validation, test)
'''
parser = ArgumentParser()
parser.add_argument('dir_food_data', help='Food dataset directory')
parser.add_argument('dir_output_images', help='Output images directory')
