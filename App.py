from MinutiaeExtractorWrapper import MinutiaeExtractorWrapper


def main():
    minutiae_extractor = MinutiaeExtractorWrapper()
    minutiae_extractor.get_classified_minutiae('/home/jakub/projects/minutiae-extractor/ClassifyNet/testData/img_files/93_5.tif')

if __name__ == "__main__":
    main()
