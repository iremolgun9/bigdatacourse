import json
import os
import cv2
import numpy as np
import uuid


def calculateCar(image, bbox):

    h, w, c = image.shape

    isRegularRect = True

    lpRatio = bbox[2] / bbox[3]

    if lpRatio < 2.3:
        isRegularRect = False

    width = bbox[2]
    height = bbox[3]

    if isRegularRect:

        # // retVal.x1 -= width * 1.3;
        # // retVal.x2 += width * 1.3;
        # // retVal.y1 -= height * 3.6;
        # // retVal.y2 += height;
        bbox[0] -= width + (width/4)
        bbox[2] += width * 2.5
        bbox[1] -= height * 4
        bbox[3] += height * 5

    else:

        bbox[0] -= width * 3.4
        bbox[2] += width * 6.8
        bbox[1] -= height * 2.5
        bbox[3] += height * 4.1

    y2 = bbox[1] + bbox[3]
    x2 = bbox[0] + bbox[2]

    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] + bbox[3] > h:
        y2 = h-1
    if bbox[0] + bbox[2] > w:
        x2 = w-1

    cropped = image[bbox[1]:y2, bbox[0]:x2]

    return cropped


with open('/media/ekin/Media6/mmr_duzce/annotations/mmr_set6.json') as json_file:
    data = json.load(json_file)

imagespath = "/media/ekin/Media6/mmr_duzce/"
annotations = data["annotations"]
images = data["images"]
categories = data['categories']
outpath = "/media/ekin/Media6/mmr_duzce/trainset/"
classes = ['alfa-romeo_Car', 'alfa-romeo_HVT', 'alfa-romeo_Van-LGT', 'alfa-romeo_Bus', 'alfa-romeo_SUV', 'audi_Car', 'audi_HVT', 'audi_Van-LGT', 'audi_Bus', 'audi_SUV', 'aston-martin_Car', 'aston-martin_HVT', 'aston-martin_Van-LGT', 'aston-martin_Bus', 'aston-martin_SUV', 'bmc_Car', 'bmc_HVT', 'bmc_Van-LGT', 'bmc_Bus', 'bmc_SUV', 'bmw_Car', 'bmw_HVT', 'bmw_Van-LGT', 'bmw_Bus', 'bmw_SUV', 'cherry_Car', 'cherry_HVT', 'cherry_Van-LGT', 'cherry_Bus', 'cherry_SUV', 'chevrolet_Car', 'chevrolet_HVT', 'chevrolet_Van-LGT', 'chevrolet_Bus', 'chevrolet_SUV', 'citroen_Car', 'citroen_HVT', 'citroen_Van-LGT', 'citroen_Bus', 'citroen_SUV', 'chrysler_Car', 'chrysler_HVT', 'chrysler_Van-LGT', 'chrysler_Bus', 'chrysler_SUV', 'daf_Car', 'daf_HVT', 'daf_Van-LGT', 'daf_Bus', 'daf_SUV', 'dacia_Car', 'dacia_HVT', 'dacia_Van-LGT', 'dacia_Bus', 'dacia_SUV', 'daewoo_Car', 'daewoo_HVT', 'daewoo_Van-LGT', 'daewoo_Bus', 'daewoo_SUV', 'daihatsu_Car', 'daihatsu_HVT', 'daihatsu_Van-LGT', 'daihatsu_Bus', 'daihatsu_SUV', 'desoto_Car', 'desoto_HVT', 'desoto_Van-LGT', 'desoto_Bus', 'desoto_SUV', 'ds-automobiles_Car', 'ds-automobiles_HVT', 'ds-automobiles_Van-LGT', 'ds-automobiles_Bus', 'ds-automobiles_SUV', 'dodge_Car', 'dodge_HVT', 'dodge_Van-LGT', 'dodge_Bus', 'dodge_SUV', 'ferrari_Car', 'ferrari_HVT', 'ferrari_Van-LGT', 'ferrari_Bus', 'ferrari_SUV', 'fiat_Car', 'fiat_HVT', 'fiat_Van-LGT', 'fiat_Bus', 'fiat_SUV', 'ford_Car', 'ford_HVT', 'ford_Van-LGT', 'ford_Bus', 'ford_SUV', 'fargo_Car', 'fargo_HVT', 'fargo_Van-LGT', 'fargo_Bus', 'fargo_SUV', 'gmc_Car', 'gmc_HVT', 'gmc_Van-LGT', 'gmc_Bus', 'gmc_SUV', 'geely_Car', 'geely_HVT', 'geely_Van-LGT', 'geely_Bus', 'geely_SUV', 'hino_Car', 'hino_HVT', 'hino_Van-LGT', 'hino_Bus', 'hino_SUV', 'honda_Car', 'honda_HVT', 'honda_Van-LGT', 'honda_Bus', 'honda_SUV', 'hyundai_Car', 'hyundai_HVT', 'hyundai_Van-LGT', 'hyundai_Bus', 'hyundai_SUV', 'infiniti_Car', 'infiniti_HVT', 'infiniti_Van-LGT', 'infiniti_Bus', 'infiniti_SUV', 'isuzu_Car', 'isuzu_HVT', 'isuzu_Van-LGT', 'isuzu_Bus', 'isuzu_SUV', 'iveco_Car', 'iveco_HVT', 'iveco_Van-LGT', 'iveco_Bus', 'iveco_SUV', 'jeep_Car', 'jeep_HVT', 'jeep_Van-LGT', 'jeep_Bus', 'jeep_SUV', 'jaguar_Car', 'jaguar_HVT', 'jaguar_Van-LGT', 'jaguar_Bus', 'jaguar_SUV', 'kia_Car', 'kia_HVT', 'kia_Van-LGT', 'kia_Bus', 'kia_SUV', 'karsan_Car', 'karsan_HVT', 'karsan_Van-LGT', 'karsan_Bus', 'karsan_SUV', 'lada_Car', 'lada_HVT', 'lada_Van-LGT', 'lada_Bus', 'lada_SUV', 'lamborghini_Car', 'lamborghini_HVT', 'lamborghini_Van-LGT', 'lamborghini_Bus', 'lamborghini_SUV', 'land-rover_Car', 'land-rover_HVT', 'land-rover_Van-LGT', 'land-rover_Bus', 'land-rover_SUV', 'lancia_Car', 'lancia_HVT', 'lancia_Van-LGT', 'lancia_Bus', 'lancia_SUV', 'lexus_Car', 'lexus_HVT', 'lexus_Van-LGT', 'lexus_Bus', 'lexus_SUV', 'mazda_Car', 'mazda_HVT', 'mazda_Van-LGT', 'mazda_Bus', 'mazda_SUV', 'mercedes_Car', 'mercedes_HVT', 'mercedes_Van-LGT', 'mercedes_Bus', 'mercedes_SUV', 'mini_Car', 'mini_HVT', 'mini_Van-LGT', 'mini_Bus', 'mini_SUV', 'mitsubihi_Car', 'mitsubihi_HVT', 'mitsubihi_Van-LGT', 'mitsubihi_Bus', 'mitsubihi_SUV', 'man_Car', 'man_HVT', 'man_Van-LGT', 'man_Bus', 'man_SUV', 'maserati_Car', 'maserati_HVT', 'maserati_Van-LGT', 'maserati_Bus', 'maserati_SUV', 'nissan_Car', 'nissan_HVT', 'nissan_Van-LGT', 'nissan_Bus', 'nissan_SUV', 'opel_Car', 'opel_HVT', 'opel_Van-LGT', 'opel_Bus', 'opel_SUV', 'otokar_Car', 'otokar_HVT', 'otokar_Van-LGT', 'otokar_Bus', 'otokar_SUV', 'peugeot_Car', 'peugeot_HVT', 'peugeot_Van-LGT', 'peugeot_Bus', 'peugeot_SUV', 'porsche_Car', 'porsche_HVT', 'porsche_Van-LGT', 'porsche_Bus', 'porsche_SUV', 'proton_Car', 'proton_HVT', 'proton_Van-LGT', 'proton_Bus', 'proton_SUV', 'renault_Car', 'renault_HVT', 'renault_Van-LGT', 'renault_Bus', 'renault_SUV', 'rover_Car', 'rover_HVT', 'rover_Van-LGT', 'rover_Bus', 'rover_SUV', 'saab_Car', 'saab_HVT', 'saab_Van-LGT', 'saab_Bus', 'saab_SUV', 'seat_Car', 'seat_HVT', 'seat_Van-LGT', 'seat_Bus', 'seat_SUV', 'skoda_Car', 'skoda_HVT', 'skoda_Van-LGT', 'skoda_Bus', 'skoda_SUV', 'smart_Car', 'smart_HVT', 'smart_Van-LGT', 'smart_Bus', 'smart_SUV', 'ssangyong_Car', 'ssangyong_HVT', 'ssangyong_Van-LGT', 'ssangyong_Bus', 'ssangyong_SUV', 'subaru_Car', 'subaru_HVT', 'subaru_Van-LGT', 'subaru_Bus', 'subaru_SUV', 'suzuki_Car', 'suzuki_HVT', 'suzuki_Van-LGT', 'suzuki_Bus', 'suzuki_SUV', 'scania_Car', 'scania_HVT', 'scania_Van-LGT', 'scania_Bus', 'scania_SUV', 'tofaş_Car', 'tofaş_HVT', 'tofaş_Van-LGT', 'tofaş_Bus', 'tofaş_SUV', 'toyota_Car', 'toyota_HVT', 'toyota_Van-LGT', 'toyota_Bus', 'toyota_SUV', 'tata_Car', 'tata_HVT', 'tata_Van-LGT', 'tata_Bus', 'tata_SUV', 'temsa_Car', 'temsa_HVT', 'temsa_Van-LGT', 'temsa_Bus', 'temsa_SUV', 'tesla_Car', 'tesla_HVT', 'tesla_Van-LGT', 'tesla_Bus', 'tesla_SUV', 'volkswagen_Car', 'volkswagen_HVT', 'volkswagen_Van-LGT', 'volkswagen_Bus', 'volkswagen_SUV', 'volvo_Car', 'volvo_HVT', 'volvo_Van-LGT', 'volvo_Bus', 'volvo_SUV', 'gaz_Car', 'gaz_HVT', 'gaz_Van-LGT', 'gaz_Bus', 'gaz_SUV', 'foton_Car', 'foton_HVT', 'foton_Van-LGT', 'foton_Bus', 'foton_SUV', 'dfsk_Car', 'dfsk_HVT', 'dfsk_Van-LGT', 'dfsk_Bus', 'dfsk_SUV', 'neoplan_Car', 'neoplan_HVT', 'neoplan_Van-LGT', 'neoplan_Bus', 'neoplan_SUV']
for typeClass in classes:
    if not os.path.exists(os.path.join(outpath, typeClass)):
        os.makedirs(os.path.join(outpath, typeClass))
for i in annotations:
    image = images[i['image_id']-1]
    imageName = os.path.basename(image['file_name'])
    if 6 <= int(imageName.split("_")[0]) <= 10:
        if int(imageName.split("_")[0]) == 10:
            if int(imageName.split("_")[1]) > 59:
                continue
            if float(imageName.split("_")[2]) > 56.000 and int(imageName.split("_")[1]) == 59:
                continue


        img = cv2.imread(imagespath + image['file_name'])
        bbox = np.array(i['bbox'], dtype=int)
        att = i['attributes']
        marka = att['mark']
        classname = ""
        if i["category_id"] == 2 or i["category_id"] == 3 or i["category_id"] == 8 or i["category_id"] == 9:
            classname = marka + "_Car"
        if i["category_id"] == 4 or i["category_id"] == 7 or i["category_id"] == 1:
            classname = marka + "_Van-LGT"
        if i["category_id"] == 5:
            classname = marka + "_HVT"
        if i["category_id"] == 10:
            classname = marka + "_SUV"
        if i["category_id"] == 6:
            classname = marka + "_Bus"
        cropped = calculateCar(img,bbox)
        # cropped = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        if classname is not "":
            cv2.imwrite(outpath + classname + "/" + str(uuid.uuid4()) + ".jpg", cropped)


    if 13 <= int(imageName.split("_")[0]) <= 14:
        if int(imageName.split("_")[0]) == 14:
            if int(imageName.split("_")[1]) > 49:
                continue
            if float(imageName.split("_")[2]) > 49.000 and int(imageName.split("_")[1]) == 49:
                continue
        if int(imageName.split("_")[0]) == 13:
            if int(imageName.split("_")[1]) < 10:
                continue
            if float(imageName.split("_")[2]) < 45.000 and int(imageName.split("_")[1]) == 10:
                continue

        img = cv2.imread(imagespath + image['file_name'])
        bbox = np.array(i['bbox'], dtype=int)
        att = i['attributes']
        marka = att['mark']
        classname = ""
        if i["category_id"] == 2 or i["category_id"] == 3 or i["category_id"] == 8 or i["category_id"] == 9:
            classname = marka + "_Car"
        if i["category_id"] == 4 or i["category_id"] == 7 or i["category_id"] == 1:
            classname = marka + "_Van-LGT"
        if i["category_id"] == 5:
            classname = marka + "_HVT"
        if i["category_id"] == 10:
            classname = marka + "_SUV"
        if i["category_id"] == 6:
            classname = marka + "_Bus"
        cropped = calculateCar(img, bbox)
        # cropped = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        if classname is not "":
            cv2.imwrite(outpath + classname + "/" + str(uuid.uuid4()) + ".jpg", cropped)

    if 16 <= int(imageName.split("_")[0]) <= 18:
        if int(imageName.split("_")[0]) == 18:
            if int(imageName.split("_")[1]) > 30:
                continue
            if float(imageName.split("_")[2]) > 32.000 and int(imageName.split("_")[1]) == 30:
                continue
        if int(imageName.split("_")[0]) == 16:
            if int(imageName.split("_")[1]) < 1:
                continue
            if float(imageName.split("_")[2]) < 43.000 and int(imageName.split("_")[1]) == 1:
                continue

        img = cv2.imread(imagespath + image['file_name'])
        bbox = np.array(i['bbox'], dtype=int)
        att = i['attributes']
        marka = att['mark']
        classname = ""
        if i["category_id"] == 2 or i["category_id"] == 3 or i["category_id"] == 8 or i["category_id"] == 9:
            classname = marka + "_Car"
        if i["category_id"] == 4 or i["category_id"] == 7 or i["category_id"] == 1:
            classname = marka + "_Van-LGT"
        if i["category_id"] == 5:
            classname = marka + "_HVT"
        if i["category_id"] == 10:
            classname = marka + "_SUV"
        if i["category_id"] == 6:
            classname = marka + "_Bus"
        cropped = calculateCar(img, bbox)
        # cropped = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        if classname is not "":
            cv2.imwrite(outpath + classname + "/" + str(uuid.uuid4()) + ".jpg", cropped)