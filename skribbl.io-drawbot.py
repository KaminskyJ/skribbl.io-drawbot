# misc
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Reading screen/grabbing text
from mss import mss
import cv2
import pytesseract as pt

# image grabbing and processing
from google_images_download import google_images_download
from simplification.cutil import simplify_coords
from skimage import color
from skimage import measure

#for drawing
import pyautogui as pa


def screen():
    """
    returns image of cropped screen containing important game prompts
    """
    with mss() as sct:
        img = sct.shot()
    img = cv2.imread(img)
    img = img[990:1200, 1050:2030]  # Parameters for given window
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def goodify(contours):
    """
    Sort countours in terms of length (importance). Additional code could be
    added here such as a randomizer to make the drawing more human and imperfect.
    """
    contours.sort(key=(lambda x: len(x)), reverse=True)
    # for all contours -> add noise
    return contours


def draw_test(contours):
    """
    function used to test and view contour plots. Replace draw() with draw_test()
    to output plots instead of issuing drawing commands.
    """
    for n, contour in enumerate(contours):
        plt.plot(368 + contour[:, 1], 250 + contour[:, 0], linewidth=2, c="k")
    plt.axis([0, 1680, 1050, 0])
    plt.show()


def draw(contours):
    """
    This function controls your mouse and draws the calculated contours

    Move mouse quickly to the top-left to stop this function
    """
    t0 = time.time()
    for n, contour in enumerate(contours):
        if time.time() - t0 < 80:
            # simplify contours for faster and easier drawing
            contour = simplify_coords(contour, 2.0)

            # 468 and 250 are x,y coordinates for top-left of the drawing space
            pa.moveTo(contour[0][1] + 468, contour[0][0] + 250)

            # draw as much as possible before 80 seconds!
            # to make it look more human, add pauses and speed randomizers
            for x in contour[1:]:
                if time.time() - t0 < 80:
                    pa.dragTo(x[1] + 468, x[0] + 250, 0)
        else:
            break


def find_best_word(choices):
    """
    For each word, finds image and calculates contours at various thresholds.
    Then takes the "best" image given metrics such as number of contours and
    contour lengths

    returns data for "best" contour
    """
    contours = []
    imgs = []
    for word in choices:

        # query google to find images
        response = google_images_download.googleimagesdownload()
        arguments = {
            "keywords": word,
            "limit": 1,
            "print_urls": False,
            "no-download": True,
            "type": "line-drawing"
        }  #creating list of arguments
        paths = response.download(arguments)
        img = mpimg.imread(paths.get(word)[0])
        black = color.rgb2gray(img)
        imgs += [black]

        # arrange and process contours
        contours1 = goodify(measure.find_contours(black, 0.1, "high"))
        contours2 = goodify(measure.find_contours(black, 0.8, "high"))

        if (len(contours1) < len(contours2) and len(contours1) > 100):
            contours += [contours1[:300]]
        else:
            contours += [contours2[:300]]
    lengths = []
    for x in range(3):
        length = 0
        for contour in contours[x][:100]:
            length += contour.shape[0]
        lengths += [length]

    best_ind = lengths.index(max(lengths))
    w, h = imgs[best_ind].shape[::-1]
    best_contour = contours[best_ind]
    for n, contour in enumerate(best_contour):
        contour *= 500 / max(w, h)  # max image size will be 500x500

    return best_contour, best_ind


def begin_turn(choices, coords):
    """
    Calls the find_best_word function, and then clicks on the best choice.
    """
    choice, ind = find_best_word(choices)
    # x,y constants will be different depending on window size
    pa.click(x=495 + (coords[2 * ind] + coords[2 * ind + 1]) // 4, y=575)
    draw(choice)


def get_imgs(img):
    """
    Splits game prompt image into separate images with a single word in each
    for easy image to text classification
    """
    means = np.mean(img, 0)
    coords = []
    for x in range(len(means) - 1):
        if abs(means[x] - means[x + 1]) > 130:
            coords += [x + 1]

    imgs = [
        img[:, coords[0] + 5:coords[1] - 5],
        img[:, coords[2] + 5:coords[3] - 5],
        img[:, coords[4] + 5:coords[5] - 5]
    ]
    return imgs, coords


def main():
    """
    Runs in the background until user triggers the failsafe.

    Continually checks the screen until it sees the game indicator to start the 
    turn. Then the turn begins.
    """
    keep_running = True
    while (keep_running):
        try:
            # config for image to text
            config = ('-l eng --oem 1 --psm 3')
            pic = screen()
            text = pt.image_to_string(
                pic[:100][:], config=config)  # changes on window size
            if text.find("Choose") != -1:
                imgs, coords = get_imgs(
                    pic[120:190][:])  # changes on window size
                word1 = pt.image_to_string(imgs[0], config=config)
                word2 = pt.image_to_string(imgs[1], config=config)
                word3 = pt.image_to_string(imgs[2], config=config)
                begin_turn([word1, word2, word3], coords)
        except pa.FailSafeException:
            # moving cursor to top-left triggers the failsafe
            keep_running = False


if __name__ == "__main__":
    main()
