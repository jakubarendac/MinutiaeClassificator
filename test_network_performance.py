# ----- CONFUSION MATRIX -----

# from keras.optimizers import Adam

# from ClassifyNet.ClassifyNet_model import ClassifyNetModel
# from ClassifyNet.ClassifyNet_utils import get_confusion_matrix
# from ClassifyNet.ClassifyNet_constants import NUM_CLASSES, INPUT_SHAPE

# classify_net_model_path = '/home/jakub/projects/minutiae-extractor/models/ClassifyNet_30_4.h5'
# test_data_path = '/home/jakub/projects/classifyNet_trainDataset/new/validate/'

# classify_net_model = ClassifyNetModel(NUM_CLASSES, classify_net_model_path, INPUT_SHAPE)
# classify_net_model.compile(loss='categorical_crossentropy',
#                                    optimizer=Adam(lr=0),
#                                    metrics=['accuracy'])

# get_confusion_matrix(classify_net_model, test_data_path)

# ----- SPEED PERFORMANCE -----

from __future__ import division
import os
from datetime import datetime

from MinutiaeExtractorWrapper import MinutiaeExtractorWrapper

IMAGE_FOLDER = "/home/jakub/projects/biometric DBs/FVC_Fingerprint_DB/FVC2004/test/"

minutiae_classificator = MinutiaeExtractorWrapper()

minutiae_classificator.get_coarse_net_path('/home/jakub/projects/minutiae-extractor/models/CoarseNet.h5')
minutiae_classificator.get_fine_net_path('/home/jakub/projects/minutiae-extractor/models/FineNet.h5')
minutiae_classificator.get_classify_net_path('/home/jakub/projects/minutiae-extractor/models/ClassifyNet.h5')

minutiae_classificator.load_extraction_module()
minutiae_classificator.load_classification_module()

date_time = datetime.now().strftime('%Y%m%d-%H%M%S')

output_file_name = 'network_speed_performance_test_' + date_time + '.txt'

total_extraction_time = 0
total_classification_time = 0
extracted_images_count = 0
classified_images_count = 0
extracted_minutiae_count = 0
classified_minutiae_count = 0

output_file = open(output_file_name, 'a')

for subdir, dirs, files in os.walk(IMAGE_FOLDER):
    for file_name in files:
        file_path = IMAGE_FOLDER + file_name

        extracted_minutiae_data = minutiae_classificator.get_extracted_minutiae(file_path, as_image = False, should_get_time = True)
        classified_minutiae_data = minutiae_classificator.get_classified_minutiae(file_path, as_image = False, should_get_time = True)

        output_file.write('File: ' + file_name + '\n')

        if extracted_minutiae_data:
            total_extraction_time += extracted_minutiae_data['time_elapsed']
            extracted_minutiae_count += extracted_minutiae_data['minutiae'].shape[0]
            extracted_images_count += 1

            print 'extracted data -> ', extracted_minutiae_data['time_elapsed'], extracted_minutiae_data['minutiae'].shape[0]
            output_file.write('\textraction_time: ' + str(extracted_minutiae_data['time_elapsed']) + ' sec.\n')
            output_file.write('\textracted_minutiae_count: ' + str(extracted_minutiae_data['minutiae'].shape[0]) + '\n\n')
        
        else:
            print 'error occured during extraction'
            output_file.write('\terror occured during extraction - no minutiae extracted\n')

        if classified_minutiae_data:
            total_classification_time += classified_minutiae_data['time_elapsed']
            classified_minutiae_count += classified_minutiae_data['minutiae'].shape[0]
            classified_images_count += 1

            print 'classified data -> ', classified_minutiae_data['time_elapsed'], classified_minutiae_data['minutiae'].shape[0]
            output_file.write('\textraction_time + classification_time: ' + str(classified_minutiae_data['time_elapsed']) + ' sec.\n')
            output_file.write('\tclassified_minutiae_count: ' + str(classified_minutiae_data['minutiae'].shape[0]) + '\n\n')

        else:
            print 'error occured during extraction + classification'
            output_file.write('\terror occured during extraction/classification - no minutiae extracted + classified\n\n')

average_extraction_time_minutiae = total_extraction_time / extracted_minutiae_count if extracted_minutiae_count > 0 else 0
average_classification_time_minutiae = total_classification_time / classified_minutiae_count if classified_minutiae_count > 0 else 0

average_extraction_time_file = total_extraction_time / extracted_images_count if extracted_images_count > 0 else 0
average_classification_time_file = total_classification_time / classified_images_count if classified_images_count > 0 else 0

output_file.write('Total extracted images count: ' + str(extracted_images_count) + '\n')
output_file.write('Total extracted + classified images count: ' + str(classified_images_count) + '\n')
output_file.write('Total extracted minutiae: ' + str(extracted_minutiae_count) + '\n')
output_file.write('Total extracted + classified minutiae: ' + str(classified_minutiae_count) + '\n')
output_file.write('Total extraction time: ' + str(total_extraction_time) + ' sec.\n')
output_file.write('Total extraction + classification time: ' + str(total_classification_time) + ' sec.\n')
output_file.write('Average extraction time for 1 file: ' + str(average_extraction_time_file) + ' sec.\n')
output_file.write('Average extraction + classification time for 1 file: ' + str(average_classification_time_file) + ' sec.\n')
output_file.write('Average extraction time for 1 minutiae: ' + str(average_extraction_time_minutiae) + ' sec.\n')
output_file.write('Average extraction + classification time for 1 minutiae: ' + str(average_classification_time_minutiae) + ' sec.\n')
    
output_file.close()

print 'preslo cele'