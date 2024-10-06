import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def seed_setup(ran_seed=42):
    random.seed(ran_seed)
    np.random.seed(ran_seed)

def function_otsu(picture, thresholds):
    pixel_sum = picture.size
    histogram, bins = np.histogram(picture.flatten(), bins=256, range=[0, 256])
    
    thresholds = [0] + thresholds + [255]
    class_variance_between = 0
    
    for x in range(1, len(thresholds)):
        up = thresholds[x]
        low = thresholds[x-1]
        
        hist_area = histogram[low:up]  #slicing histogram
        pixel_area = np.sum(hist_area)
        area_weight = pixel_area / pixel_sum
        
        if pixel_area == 0:
            continue  #move on if there is pixels in that area
        #calculating the mean intensity for the region
        area_mean = np.sum(hist_area * np.arange(low, up)) / pixel_area
        #calculating the total mean for the image
        total_mean = np.sum(histogram * np.arange(256)) / pixel_sum
        
        class_variance_between += area_weight * (area_mean - total_mean) ** 2
    
    return class_variance_between


def function_kapur(picture, thresholds):
    pixel_sum = picture.size
    histogram, bins = np.histogram(picture.flatten(), bins=256, range=[0, 256])
    
    thresholds = [0] + thresholds + [255]
    final_entropy = 0
    
    for x in range(1, len(thresholds)):
        up = thresholds[x]
        low = thresholds[x-1]
        
        hist_area = histogram[low:up]
        pixel_area = np.sum(hist_area)
        
        if pixel_area == 0:
            continue  #skipping if no pixels in the region
        
        #the probalibilty distribution is calculated
        probability_distribution = hist_area / pixel_area
        probability_distribution = probability_distribution[probability_distribution > 0]  #we do this to avoid log(0)
        
        #the entropy is calculated
        entropy_area = -np.sum(probability_distribution * np.log(probability_distribution))
        final_entropy += entropy_area * (pixel_area / pixel_sum)
    
    return final_entropy

def simulated_annealing(picture, objective_function, initial_thresholds, max_iteration=1000,ran_seed=42):
    seed_setup(ran_seed)
    threshold_now = initial_thresholds.copy()
    threshold_best = threshold_now.copy()
    score_best = objective_function(picture, threshold_best)
    
    cool_rate = 0.99
    temp = 100.0
    
    for x in range(max_iteration):
        #small random tweak to the thresholds
        threshold_new = [y + random.choice([-1, 1]) for y in threshold_now]
        threshold_new = sorted([max(0, min(255, z)) for z in threshold_new])  #ensure valid thresholds
        
        score_new = objective_function(picture, threshold_new)
        
        # accept if better, or with some probability if worse
        if score_new > score_best or random.random() < np.exp((score_new - score_best) / temp):
            threshold_now = threshold_new
            score_best = score_new
            threshold_best = threshold_now.copy()
        
        
        temp *= cool_rate # decreasing the temperature
    
    return score_best, threshold_best


def variable_neighbourhood_search(picture, objective_function, initial_thresholds, max_iteration=1000,ran_seed=42):
    seed_setup(ran_seed)
    threshold_now = initial_thresholds.copy()
    threshold_best = threshold_now.copy()
    score_best = objective_function(picture, threshold_best)
    
    for x in range(max_iteration):
        if x % 2 == 0:
            threshold_new = [y + random.choice([-1, 1]) for y in threshold_now] #changing a bit locally
        else:
            threshold_new = [y + random.choice([-10, 10]) for y in threshold_now] #large change globally
        
        threshold_new = sorted([max(0, min(255, t)) for t in threshold_new])
        score_new = objective_function(picture, threshold_new)
        
        
        if score_new > score_best: #set new bes if the new solution we got is way better
            threshold_now = threshold_new
            score_best = score_new
            threshold_best = threshold_now.copy()
    
    return score_best, threshold_best


def apply_thresholds(picture, thresholds):  #this is a function that applies our thresholds to an image and segment it
    thresholds = [0] + thresholds + [255]  #add 0 and 255 as boundaries
    segmented_picture = np.zeros_like(picture)
    
    colors = [
        [255, 0, 0],   #R
        [0, 255, 0],   #G
        [0, 0, 255],   #B
        [255, 255, 0], #Y
        [255, 0, 255], #M
        [0, 255, 255]  #C 
    ]
    
    for i in range(1, len(thresholds)):
        lower = thresholds[i-1]
        upper = thresholds[i]
        mask = (picture >= lower) & (picture < upper)
        segmented_picture[mask] = (lower + upper) // 2  #give a mean of threshold as segment label
    
    return segmented_picture

def create_histogram(picture, ax):
    standard_dev_pixel = np.std(picture)
    range_pixel = picture.max() - picture.min()
    
    bins = int(min(64, max(16, standard_dev_pixel)))
    
    ax.hist(picture.ravel(), bins=bins, range=[0, 256], color='black', density=True)
    ax.set_title('Histogram')
    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')


def show_segmented_images(original_picture, sa_otsu, vns_otsu, sa_kapur, vns_kapur):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    axs[0, 0].imshow(original_picture, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(sa_otsu)
    axs[0, 1].set_title("SA Otsu Segmentation")
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(vns_otsu)
    axs[0, 2].set_title("VNS Otsu Segmentation")
    axs[0, 2].axis('off')

    axs[0, 3].imshow(sa_kapur)
    axs[0, 3].set_title("SA Kapur Segmentation")
    axs[0, 3].axis('off')
    
    axs[1, 1].imshow(vns_kapur)
    axs[1, 1].set_title("VNS Kapur Segmentation")
    axs[1, 1].axis('off')
    
    create_histogram(original_picture, axs[1, 0])

    axs[1, 2].axis('off')
    axs[1, 3].axis('off')

    plt.tight_layout()
    plt.show()

def individual_image(pic_path, final_folder='Code/output_images',ran_seed=42):
    seed_setup(ran_seed)
    picture = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    initial_thresholds = [2, 3, 4, 5]

    sa_otsu_score, sa_otsu_thresholds = simulated_annealing(picture, function_otsu, initial_thresholds)
    vns_otsu_score, vns_otsu_thresholds = variable_neighbourhood_search(picture, function_otsu, initial_thresholds)

    sa_kapur_score, sa_kapur_thresholds = simulated_annealing(picture, function_kapur, initial_thresholds)
    vns_kapur_score, vns_kapur_thresholds = variable_neighbourhood_search(picture, function_kapur, initial_thresholds)

    sa_otsu_segmented = apply_thresholds(picture, sa_otsu_thresholds)
    vns_otsu_segmented = apply_thresholds(picture, vns_otsu_thresholds)
    sa_kapur_segmented = apply_thresholds(picture, sa_kapur_thresholds)
    vns_kapur_segmented = apply_thresholds(picture, vns_kapur_thresholds)
    
    base_name = os.path.basename(pic_path).split('.')[0]
    cv2.imwrite(os.path.join(final_folder, f"{base_name}_sa_otsu.png"), sa_otsu_segmented)
    cv2.imwrite(os.path.join(final_folder, f"{base_name}_vns_otsu.png"), vns_otsu_segmented)
    cv2.imwrite(os.path.join(final_folder, f"{base_name}_sa_kapur.png"), sa_kapur_segmented)
    cv2.imwrite(os.path.join(final_folder, f"{base_name}_vns_kapur.png"), vns_kapur_segmented)
    
    ssim_sa_otsu = ssim(picture, sa_otsu_segmented, data_range=picture.max() - picture.min())
    psnr_sa_otsu = psnr(picture, sa_otsu_segmented, data_range=picture.max() - picture.min())

    ssim_vns_otsu = ssim(picture, vns_otsu_segmented, data_range=picture.max() - picture.min())
    psnr_vns_otsu = psnr(picture, vns_otsu_segmented, data_range=picture.max() - picture.min())


    show_segmented_images(picture, sa_otsu_segmented, vns_otsu_segmented, sa_kapur_segmented, vns_kapur_segmented)

    return {
        'sa_otsu': (sa_otsu_thresholds, ssim_sa_otsu, psnr_sa_otsu),
        'vns_otsu': (vns_otsu_thresholds, ssim_vns_otsu, psnr_vns_otsu),
        'sa_kapur': (sa_kapur_thresholds,), 
        'vns_kapur': (vns_kapur_thresholds,),  
    }



def go_all_folder(pics_folder, final_folder='Code/output_images',ran_seed=42):
    os.makedirs(final_folder, exist_ok=True)
    seed_setup(ran_seed)
    results = {}  

    for picture_name in os.listdir(pics_folder):
        if picture_name.endswith((".png", ".jpg", ".jpeg")):
            pic_path = os.path.join(pics_folder, picture_name)
            results[picture_name] = individual_image(pic_path)

 
    print("Image\tLevel\tOtsu (SA/VNS)\t\tKapur (SA/VNS)\t\tSSIM (SA/VNS)\t\tPSNR (SA/VNS)")
    print("-" * 100)  

    for image_name, image_data in results.items():
        for level in range(2, 6): 
            print(f"{image_name}\t{level}\t", end="")

            
            print(f"{image_data['sa_otsu'][0][level-2]:.2f}/{image_data['vns_otsu'][0][level-2]:.2f}\t\t", end="")
            print(f"{image_data['sa_kapur'][0][level-2]:.2f}/{image_data['vns_kapur'][0][level-2]:.2f}\t\t", end="")
        
            print(f"{image_data['sa_otsu'][1]:.4f}/{image_data['vns_otsu'][1]:.4f}\t\t", end="")
            print(f"{image_data['sa_otsu'][2]:.4f}/{image_data['vns_otsu'][2]:.4f}") 

        print()  

folder_path = 'Code/Ass2'
go_all_folder(folder_path)
