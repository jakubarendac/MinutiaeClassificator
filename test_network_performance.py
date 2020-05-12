# ----- NETWORK PERFORMANCE TESTS -----
# This file contains performance tests for our library:
# Test n.1 - confusion matrix of ClassifyNet
# Test n.2 - performance speed test on current device

# ----- CONFUSION MATRIX -----
# import os
# import numpy as np

# from ClassifyNetWrapper import ClassifyNetWrapper
# from ClassifyNet.ClassifyNet_utils import getMinutiaeTypeFromId
# from ClassifyNet.ClassifyNet_model import plot_confusion_matrix
# from ClassifyNet.ClassifyNet_constants import MINUTIAE_CLASSES
# from utils.image_utils import read_image

# classify_net_model_path = '/home/jakub/projects/minutiae-extractor/models/ClassifyNet_6_classes.h5'
# test_data_path = '/home/jakub/projects/classifyNet_trainDataset/new/validate/'

# classifyNet = ClassifyNetWrapper(classify_net_model_path)

# ending = []
# bifurcation = []
# fragment = []
# enclosure = []
# crossbar = []
# other = []

# for subdir, dirs, files in os.walk(test_data_path):
#     for dir in dirs:
#         for subdir, dirs, files in os.walk(test_data_path + dir):
#             end = 0
#             bif = 0
#             fra = 0
#             enc = 0
#             cro = 0
#             oth = 0

#             for file_name in files:
#                 file_path = test_data_path + dir + '/' + file_name

#                 image = read_image(file_path)
#                 classified = classifyNet.classify_minutiae_patch(image['original_image'])
#                 classified_type = getMinutiaeTypeFromId(classified)

#                 if classified_type is 'ending':
#                     end += 1
#                 if classified_type is 'bifurcation':
#                     bif += 1
#                 if classified_type is 'fragment':
#                     fra += 1
#                 if classified_type is 'enclosure':
#                     enc += 1
#                 if classified_type is 'crossbar':
#                     cro += 1
#                 if classified_type is 'other':
#                     oth += 1

#             if dir == 'ending':
#                 ending = [end, bif, fra, enc, cro, oth]
#             if dir == 'bifurcation':
#                 bifurcation = [end, bif, fra, enc, cro, oth]
#             if dir == 'fragment':
#                 fragment = [end, bif, fra, enc, cro, oth]
#             if dir == 'enclosure':
#                 enclosure = [end, bif, fra, enc, cro, oth]
#             if dir == 'crossbar':
#                 crossbar = [end, bif, fra, enc, cro, oth]
#             if dir == 'other':
#                 other = [end, bif, fra, enc, cro, oth]

# predictions = np.array([ending, bifurcation, fragment, enclosure, crossbar, other])

# plot_confusion_matrix(predictions, MINUTIAE_CLASSES, title='Confusion matrix - validation dataset', save_image=True)

# ----- SPEED PERFORMANCE -----

# from __future__ import division
# import os
# from datetime import datetime

# from MinutiaeExtractorWrapper import MinutiaeExtractorWrapper

# IMAGE_FOLDER = "/home/jarendac/projects/biometric_DBs/FVC/FVC2004/DB1_A/"

# minutiae_classificator = MinutiaeExtractorWrapper()

# minutiae_classificator.get_coarse_net_path('/home/jarendac/projects/MinutiaeClassificator/MinutiaeNet/Models/CoarseNet.h5')
# minutiae_classificator.get_fine_net_path('/home/jarendac/projects/MinutiaeClassificator/MinutiaeNet/Models/FineNet.h5')
# minutiae_classificator.get_classify_net_path('/home/jarendac/projects/MinutiaeClassificator/ClassifyNet/output_ClassifyNet/20200430-004944/ClassifyNet_patch224batch32_model.h5')

# minutiae_classificator.load_extraction_module()
# minutiae_classificator.load_classification_module()

# date_time = datetime.now().strftime('%Y%m%d-%H%M%S')

# output_file_name = 'network_speed_performance_test_' + date_time + '.txt'

# total_extraction_time = 0
# total_classification_time = 0
# extracted_images_count = 0
# classified_images_count = 0
# extracted_minutiae_count = 0
# classified_minutiae_count = 0

# output_file = open(output_file_name, 'a')

# for subdir, dirs, files in os.walk(IMAGE_FOLDER):
#     for file_name in files:
#         file_path = IMAGE_FOLDER + file_name

#         extracted_minutiae_data = minutiae_classificator.get_extracted_minutiae(file_path, as_image = False, should_get_time = True)
#         classified_minutiae_data = minutiae_classificator.get_classified_minutiae(file_path, as_image = False, should_get_time = True)

#         output_file.write('File: ' + file_name + '\n')

#         if extracted_minutiae_data:
#             total_extraction_time += extracted_minutiae_data['time_elapsed']
#             extracted_minutiae_count += extracted_minutiae_data['minutiae'].shape[0]
#             extracted_images_count += 1

#             print 'extracted data -> ', extracted_minutiae_data['time_elapsed'], extracted_minutiae_data['minutiae'].shape[0]
#             output_file.write('\textraction_time: ' + str(extracted_minutiae_data['time_elapsed']) + ' sec.\n')
#             output_file.write('\textracted_minutiae_count: ' + str(extracted_minutiae_data['minutiae'].shape[0]) + '\n\n')
        
#         else:
#             print 'error occured during extraction'
#             output_file.write('\terror occured during extraction - no minutiae extracted\n')

#         if classified_minutiae_data:
#             total_classification_time += classified_minutiae_data['time_elapsed']
#             classified_minutiae_count += classified_minutiae_data['minutiae'].shape[0]
#             classified_images_count += 1

#             print 'classified data -> ', classified_minutiae_data['time_elapsed'], classified_minutiae_data['minutiae'].shape[0]
#             output_file.write('\textraction_time + classification_time: ' + str(classified_minutiae_data['time_elapsed']) + ' sec.\n')
#             output_file.write('\tclassified_minutiae_count: ' + str(classified_minutiae_data['minutiae'].shape[0]) + '\n\n')

#         else:
#             print 'error occured during extraction + classification'
#             output_file.write('\terror occured during extraction/classification - no minutiae extracted + classified\n\n')

# average_extraction_time_minutiae = total_extraction_time / extracted_minutiae_count if extracted_minutiae_count > 0 else 0
# average_classification_time_minutiae = total_classification_time / classified_minutiae_count if classified_minutiae_count > 0 else 0

# average_extraction_time_file = total_extraction_time / extracted_images_count if extracted_images_count > 0 else 0
# average_classification_time_file = total_classification_time / classified_images_count if classified_images_count > 0 else 0

# output_file.write('Total extracted images count: ' + str(extracted_images_count) + '\n')
# output_file.write('Total extracted + classified images count: ' + str(classified_images_count) + '\n')
# output_file.write('Total extracted minutiae: ' + str(extracted_minutiae_count) + '\n')
# output_file.write('Total extracted + classified minutiae: ' + str(classified_minutiae_count) + '\n')
# output_file.write('Total extraction time: ' + str(total_extraction_time) + ' sec.\n')
# output_file.write('Total extraction + classification time: ' + str(total_classification_time) + ' sec.\n')
# output_file.write('Average extraction time for 1 file: ' + str(average_extraction_time_file) + ' sec.\n')
# output_file.write('Average extraction + classification time for 1 file: ' + str(average_classification_time_file) + ' sec.\n')
# output_file.write('Average extraction time for 1 minutiae: ' + str(average_extraction_time_minutiae) + ' sec.\n')
# output_file.write('Average extraction + classification time for 1 minutiae: ' + str(average_classification_time_minutiae) + ' sec.\n')
    
# output_file.close()