import os, csv, cv2, array, fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io, color, filters, img_as_float64, morphology

csv.register_dialect('gnuplot_spaces', delimiter=" ")
csv.register_dialect('gnuplot_tabs', delimiter="\t")
# io.use_plugin('tifffile', 'imread')

def find_forward_point(image, row_index, column_index):
    """
    Finds the forward point in a blob along the center of an image
    :param image: ndarry|str|file, Image to look in
    :param blur_sigma: int, Amount to blur (std dev in px)
    :param noise_floor: float, Value 0-1 to consider as black
    :return: Optional[float] Horizontal position 0-1, if present
    """
    if not isinstance(image, (np.ndarray, np.generic)):
        # We can take in a raw image or file/filename.
        # if the latter, load it.
        image = io.imread(image, as_gray=True)
    elif len(image.shape) == 3:
        # If a raw image in color, make it gray
        image = color.rgb2gray(image)

    # Specifically search along the point of maximum gradient
    processing_line = row_index

    # Search right-to-left for a point above the noise floor
    image_line = image[processing_line][:]
    right_endpoint = None
    for i in reversed(range(len(image_line))):
        if image_line[i]:
            right_endpoint = i / image.shape[0]
            break

    return right_endpoint

def find_detonation_location(kept_files, transitionIntenisty):

    # Step 1: Dtermine the frame in which DDT occurs
    max_intensity = []
    min_intensity = []
    for index in range(len(kept_files)):
        filename = f"{in_dir}{kept_files[index]}"

        # Converts all read image intensities to float values between 0 and 1.
        image = img_as_float64(io.imread(filename, as_gray=True))
        max_intensity.append(image.max())
        min_intensity.append(np.min(image[np.nonzero(image)]))
        if len(max_intensity) > 2:
            difference = abs(
                (max_intensity[index] - max_intensity[index - 1]) / max_intensity[
                    index - 1])
        else:
            difference = 0
        # print(difference)
        # This stops the code from processing post detonation images
        if max_intensity[index] >= 0.99 * transitionIntenisty:
            transition_frame = index
            break
        if difference > max_intensity_change:
            transition_frame = index
            break
    # Knowing the frame of DDT, now determine the location
    transition_filename = f"{in_dir}{kept_files[transition_frame]}"
    transition_image = img_as_float64(io.imread(filename, as_gray=True))
    # Determine the gradients in luminosity of the high-speed frame
    image_gradienth = filters.sobel_h(image)
    image_gradientv = abs(filters.sobel_v(image))
    image_gradient = np.sqrt(image_gradientv ** 2 + image_gradienth ** 2)
    image_median = filters.median(image_gradient, footprint=np.ones((2, 2)))
    updated_image = image_median

    [row_index, column_index] = np.unravel_index(updated_image.argmax(), updated_image.shape)

    fig, ax = plt.subplots(2)
    ax[0].imshow(transition_image)
    ax[1].imshow(updated_image)
    plt.show()

    return transition_frame, row_index

def find_flame_location(image, output_path, filename, flame_height_frame, **kwargs):
    # Step 1: Determine prior flame location
    past_check = False
    for key, value in kwargs.items():
        if key == 'past_position':
            past_position = value
        elif key == 'past_window':
            past_window = value
        elif key == 'past_check':
            past_check = True

    # Create a smoothed intensity image
    footprint = morphology.disk(7)
    denoised_imgage = morphology.white_tophat(image, footprint)
    corrected_image = image - denoised_imgage
    image_smooth = filters.gaussian(corrected_image, sigma=1.0)
    # Determine the gradients in luminosity of the high-speed frame
    image_gradient = abs(filters.sobel_v(corrected_image))
    updated_image = image_gradient

    # Loop over number of rows in image to determine the entire flame front
    flame_span_array = np.linspace(int(64 / 2 - flame_height_frame / 2), int(64 /2 + flame_height_frame / 2), int(flame_height_frame + 1))
    flame_front_column = []
    maximum_luminosity = []

    temp_index_array = []
    for count, row in enumerate(flame_span_array):
        scaled_image = (updated_image / np.max(updated_image)) * 1
        scaled_image[scaled_image > 0.15] = 1; scaled_image[scaled_image < 0.05] = 0
        if past_check is True:
            temp_max_index = sorted(range(len(scaled_image[int(row)][int(past_position[int(count)] - past_window):])), key=lambda x: scaled_image[int(row)][int(past_position[int(count)] - past_window):][x])[-1:]
            temp_max_index = np.max(temp_max_index)
            temp_index_array.append(int(past_position[int(count)] - past_window + temp_max_index))
            # temp_index_array.append(np.argpartition(image_gradient[int(row)][int(past_position[int(count)] - past_window / 2):int(past_position[int(count)] + past_window / 2)], 10))
        else:
            temp_max_index = sorted(range(len(scaled_image[int(row)])), key=lambda x:scaled_image[int(row)][x])[-1:]
            temp_index_array.append(np.max(temp_max_index))
            # temp_index_array.append(np.argpartition(image_gradient[int(row)], 10))

    temp_index = max(temp_index_array)
    flame_front_column = temp_index_array
    maximum_luminosity.append(corrected_image[int(row)][temp_index])
    maximum_gradient = updated_image[int(row)][temp_index]
    plot_gradient_flame_profiles(output_path, filename, image, temp_index)

    fig, ax = plt.subplots(9)
    ax[0].imshow(image)
    ax[1].imshow(denoised_imgage, cmap=mpl.colormaps['gray'])
    ax[2].imshow(corrected_image, cmap=mpl.colormaps['gray'])
    ax[3].imshow(image_smooth, cmap=mpl.colormaps['gray'])
    ax[4].imshow(abs(filters.sobel(image_smooth)), cmap=mpl.colormaps['gray'])
    ax[5].imshow(abs(filters.sobel_h(image_smooth)), cmap=mpl.colormaps['gray'])
    ax[6].imshow(updated_image, cmap=mpl.colormaps['gray'])
    ax[7].imshow(scaled_image, cmap=mpl.colormaps['gray'])
    ax[7].plot(flame_front_column, flame_span_array, 'ro', markersize=1)
    ax[8].imshow(image, cmap=mpl.colormaps['gray'])
    ax[8].plot(flame_front_column, flame_span_array, 'ro', markersize=1)
    fig.savefig(f"{output_path}\\individual_luminosity_plots\\{filename}", format='tiff')
    plt.show()
    plt.close(fig)

    return flame_front_column, maximum_luminosity, maximum_gradient, np.array([corrected_image, scaled_image])

def find_flame_detonation_location(image, detonation_row, **kwargs):
    # Step 1: Determine prior flame location
    past_check = False
    for key, value in kwargs.items():
        if key == 'past_position':
            past_position = value
        elif key == 'past_window':
            past_window = value
        elif key == 'past_check':
            past_check = True

    # Create a smoothed intensity image
    footprint = morphology.disk(7)
    denoised_imgage = morphology.white_tophat(image, footprint)
    corrected_image = image - denoised_imgage
    image_smooth = filters.gaussian(corrected_image, sigma=1.0)
    # Determine the gradients in luminosity of the high-speed frame
    image_gradient = abs(filters.sobel_v(image_smooth))
    updated_image = image_gradient

    scaled_image = (updated_image / np.max(updated_image)) * 1
    scaled_image[scaled_image > 0.15] = 1; scaled_image[scaled_image < 0.05] = 0
    if past_check is True:
        temp_max_index = sorted(range(len(scaled_image[int(detonation_row), int(past_position - past_window):])), key=lambda x: scaled_image[int(detonation_row), int(past_position - past_window):][x])[-1:]
        temp_max_index = np.max(temp_max_index)
        temp_index_array = int(past_position - past_window + temp_max_index)
    else:
        temp_index_array = np.max(sorted(range(len(scaled_image[int(detonation_row), :])), key=lambda x: scaled_image[int(detonation_row), :][x])[-1:])
        temp_index_array = np.max(temp_index_array)

    temp_index = temp_index_array
    maximum_luminosity = image_smooth[detonation_row, temp_index]
    maximum_gradient = updated_image[detonation_row, temp_index]

    return temp_index, maximum_luminosity, maximum_gradient

def write_results_txt(output_dict, path, append=False):
    """
    :param output_dict: Data to be written to output file
    :param path: The file path at which the data is saved
    :param append: Option to append to a preexisting file or overwrite existing file
    default is False
    :return: filepath which is the same as the path
    """
    filepath = path

    if not os.path.exists(filepath):
        # If there isn't an existing file, we can't append to it, and
        # should write out the header.
        append = False

    mode = 'a' if append else 'w'
    with open(filepath, mode, newline='') as csvfile:
        fieldnames = list(output_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='gnuplot_spaces')

        if not append:
            writer.writeheader()

        for index in range(len(output_dict[fieldnames[0]])):
            # Big dict with lists to list of little dicts that we write
            # one-at-a-time.
            row = {name: output_dict[name][index] for name in output_dict}
            writer.writerow(row)
    return filepath

def plot_position_vs_time(input_info, save_path):
    plt.figure('position_vs_time')
    plt.title('Position Vs Time')
    plt.plot(input_info['Time'], input_info['Position'], 'o')
    plt.xlabel('Time (sec)')
    plt.ylabel('Position (m)')
    plt.savefig(save_path)
    plt.close()
    return

def plot_intensity_vs_time(input_info, save_path):
    plt.figure('luminosity_vs_time')
    plt.title('Luminosity Vs Time')
    plt.plot(input_info['Time'], input_info['Scaled Intensity'], 'o')
    plt.xlabel('Time (sec)')
    plt.ylabel('Scaled Intensity ()')
    plt.savefig(save_path)
    plt.close()
    return

def plot_position_vs_luminosity(input_info, save_path):
    plt.figure('position_vs_luminosity')
    plt.title('Position Vs Luminosity')
    plt.plot(input_info['Position'], input_info['Scaled Intensity'], 'o')
    plt.xlabel('Position (m)')
    plt.ylabel('Scaled Intensity ()')
    plt.savefig(save_path)
    plt.close()
    return

def plot_gradient_flame_profiles(output_path, filename, image, flame_location):
    footprint = morphology.disk(3)
    denoised_imgage = morphology.white_tophat(image, footprint)
    corrected_image = image - denoised_imgage
    image_gradient = abs(filters.sobel_v(corrected_image))
    if flame_location < 20:
        scaled_image = (image_gradient / max(image_gradient[32, :]))
    else:
        scaled_image = (image_gradient / max(image_gradient[32, flame_location-20:flame_location+20]))
    scaled_image[scaled_image < 0.075] = 0; scaled_image[scaled_image > 1] = 1;

    plt.figure('gradient_flame_profile')
    plt.imshow(scaled_image, cmap=mpl.colormaps['gray'])
    plt.xlabel('Pixels (px)')
    plt.savefig(f"{output_path}\\individual_flame_profiles_plots\\{filename}", format='tiff')
    plt.close()
    return


def plot_position_vs_time_evolution(output_path, filename, frame, data_array, image_array, flame_array, bounds):
    # Data Array: 1) Time 2) Position
    # Image Array: 1) raw_image 2) horizontal_grad
    # Flame Front Array: 1) flame span array 2) flame front columns
    flame_span = np.linspace(int(64 / 2 - flame_array[0] / 2), int(64 /2 + flame_array[0] / 2), int(flame_array[0] + 1))
    if plt.fignum_exists('position_vs_time_evolution'):
        plt.figure('position_vs_time_evolution')
    else:
        plt.figure('position_vs_time_evolution')
        fig, ax = plt.subplots(4, gridspec_kw={'height_ratios':[5,1,1,1]})
        ax[0].set_xlim(bounds[0][0], bounds[0][1])
        ax[0].set_ylim(bounds[1][0], bounds[1][1])

    ax[0].plot(data_array[0], data_array[1], 'k-')
    ax[0].plot(data_array[0][-1], data_array[1][-1], 'r.')
    ax[1].imshow(image_array[0], cmap=mpl.colormaps['gray'])
    ax[2].imshow(image_array[1], cmap=mpl.colormaps['gray'])
    ax[3].imshow(image_array[0], cmap=mpl.colormaps['gray'])
    ax[3].plot(flame_array[1], flame_span, 'ro', markersize=1)
    plt.suptitle('Frame: {0:5f} \nFlame Time, Position = {1:5f} s, {2:5f} m'.format(frame, data_array[0][-1], data_array[1][-1]))
    plt.savefig(f"{output_path}\\individual_position_evolution_plots\\{filename}", format='tiff')
    plt.close()
    return

def create_output_video(image_dir, video_name):
    images = [img for img in os.listdir(image_dir) if img.endswith(".tif")]
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))

    cv2.destroyAllWindows()
    video.release()
    return

if __name__ == "__main__":
    # Set Constants
    frame_rate = 160000
    meters_per_pixel = 0.00074074
    accepted_file_types = "*.png,*.jpg,*.jpeg,*.tif"
    max_intensity_change = 2.0
    flame_height_span = 24
    flame_window = 20

    # Set input Directory
    print("Importing Experimental Data")
    version_number = "V23-Nova"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    default_input_dir = dir_path+"\\Raw-Photo-Files\\Nova-Files\\"
    in_dir = default_input_dir
    # Within the input directory determine the number of folders to process
    date_folders = []
    for entry in os.listdir(default_input_dir):
        if os.path.isdir(os.path.join(default_input_dir, entry)):
            date_folders.append(entry)
    print(date_folders)

    for folder in range(len(date_folders)):
        # Set specific test input directory
        in_dir = f"{default_input_dir}{date_folders[folder]}\\"
        # Set output directory
        if os.path.exists(f"{dir_path}\\Processed-Photos-{version_number}\\") is False:
            os.mkdir(f"{dir_path}\\Processed-Photos-{version_number}\\")
        default_output_dir = f"{dir_path}\\Processed-Photos-{version_number}\\{date_folders[folder]}-output"
        out_dir = default_output_dir

        if os.path.exists(out_dir) is False:
            os.mkdir(out_dir)
        if os.path.exists(f"{out_dir}\\individual_luminosity_plots\\") is False:
            os.mkdir(f"{out_dir}\\individual_luminosity_plots\\")
        if os.path.exists(f"{out_dir}\\individual_position_evolution_plots\\") is False:
            os.mkdir(f"{out_dir}\\individual_position_evolution_plots\\")
        if os.path.exists(f"{out_dir}\\individual_flame_profiles_plots\\") is False:
            os.mkdir(f"{out_dir}\\individual_flame_profiles_plots\\")
        # Make a list of files from the wildcard glob in the config (eg, *.png)
        files = []
        globs = [g.strip() for g in accepted_file_types.split(',')]
        for file in os.listdir(in_dir):
            if any((fnmatch.fnmatch(file, glob) for glob in globs)):
                files.append(file)

        # Remove empty images
        kept_files = []
        maxIntensityArray = []
        for index in range(len(files)):
            filename = f"{in_dir}{files[index]}"
            image = io.imread(filename, as_gray=True)
            max_intensity_value = image.max() if image.dtype.kind == "f" else image.max() / np.iinfo(
                image.dtype).max
            maxIntensityArray.append(max_intensity_value)
            # This value needs to be adjusted to not remove desired files
            if max_intensity_value > 1.5e-3:
                kept_files.append(files[index])

        # Determine the frame DDT occurs in and the location
        [transition_frame, transition_row] = find_detonation_location(kept_files, np.max(maxIntensityArray))

        # Now determine the flame front
        viable_frames = array.array('i', list(range(0, len(kept_files))))
        position_data = np.empty(len(viable_frames), dtype=object)
        position_detonation_data = np.empty(len(viable_frames), dtype=object)
        luminosity_data = np.empty(len(viable_frames), dtype=object)
        luminosity_detonation_data = np.empty(len(viable_frames), dtype=object)
        luminosity_grad_data = np.empty(len(viable_frames), dtype=object)
        luminosity_grad_detonation_data = np.empty(len(viable_frames), dtype=object)
        flame_position = []; flame_time = []
        for index in range(len(viable_frames)):
            filename = f"{in_dir}{kept_files[index]}"
            # Converts all read image intensities to float values between 0 and 1.
            image = img_as_float64(io.imread(filename, as_gray=True))
            image_width = len(image[0])
            if index <= (flame_window / 2):
                [position_data[index], luminosity_data[index], luminosity_grad_data[index], image_array] = find_flame_location(image, out_dir, kept_files[index], flame_height_span)
                [position_detonation_data[index], luminosity_detonation_data[index], luminosity_grad_detonation_data[index]] = find_flame_detonation_location(image, 32)
            elif index >= (image_width - (flame_window / 2)):
                [position_data[index], luminosity_data[index], luminosity_grad_data[index], image_array] = find_flame_location(image, out_dir, kept_files[index],flame_height_span)
                [position_detonation_data[index], luminosity_detonation_data[index],luminosity_grad_detonation_data[index]] = find_flame_detonation_location(image, 32)
            else:
                [position_data[index], luminosity_data[index], luminosity_grad_data[index], image_array] = find_flame_location(image, out_dir, kept_files[index], flame_height_span, past_position = position_data[index-1], past_check=True, past_window=flame_window)
                [position_detonation_data[index], luminosity_detonation_data[index], luminosity_grad_detonation_data[index]] = find_flame_detonation_location(image, 32, past_position = position_detonation_data[index - 1], past_check=True, past_window=flame_window)

            flame_time.append(0 + index * (1 / frame_rate))
            flame_position.append(position_detonation_data[index] * meters_per_pixel)
            plot_position_vs_time_evolution(out_dir, kept_files[index], index,
                                            np.array([flame_time, flame_position]),
                                            image_array,
                                            np.array([flame_height_span, position_data[index]]),
                                            np.array([np.array([0, len(viable_frames) * (1 / frame_rate)]),
                                                      np.array([0, image_width * meters_per_pixel])]))

            if np.max(position_data[index]) >= image_width - 10 or index >= transition_frame:
                position_data = position_data[position_data != np.array(None)]
                position_detonation_data = position_detonation_data[position_detonation_data != np.array(None)]
                luminosity_detonation_data = luminosity_detonation_data[luminosity_detonation_data != np.array(None)]
                luminosity_grad_detonation_data = luminosity_grad_detonation_data[luminosity_grad_detonation_data != np.array(None)]
                break
            # if index >= transition_frame:
                # break

        delta_t = []
        for index in range(len(position_detonation_data)):
            delta_t.append(0 + index * (1 / frame_rate))
        output_position = {
                            '#Index': list(range(1, len(position_detonation_data) + 1)),
                            'Time': delta_t,
                            'Position': (np.array(position_detonation_data) * meters_per_pixel).tolist(),
                            'Scaled Intensity': luminosity_detonation_data/np.max(maxIntensityArray),
                            'Intensity Gradient': luminosity_grad_detonation_data
        }
        plot_position_vs_time(output_position, f"{out_dir}\\output_center_detonation_luminosity_position.jpeg")
        plot_intensity_vs_time(output_position, f"{out_dir}\\output_center_detonation_luminosity_intensity.jpeg")
        plot_position_vs_luminosity(output_position, f"{out_dir}\\output_center_position_intensity.jpeg")
        create_output_video(f"{out_dir}\\individual_luminosity_plots\\", f"{out_dir}\\flame_front_processing_animation.mp4")
        create_output_video(f"{out_dir}\\individual_position_evolution_plots\\", f"{out_dir}\\flame_front_position_tracking_animation.mp4")
        create_output_video(f"{out_dir}\\individual_flame_profiles_plots\\", f"{out_dir}\\flame_profile_animation.mp4")
        output_filepath = write_results_txt(output_position,
                                            f"{out_dir}\\output_luminosity_center_position.txt")

        print(f"Results save to {output_filepath}")
