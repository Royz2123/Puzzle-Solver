import image_processing.image_proc as im_proc
import image_processing.util as util
import image_processing.camera as camera
import time
import algorithmics.puzzle as puzzle
import cv2
import mechanics.mechanics_api as mechanics_api

# Exciting! actual real main
# Need to think about shceme - maybe we need thread for greedy in terms of speed
# If it's slow, then greedy and execution are intertwined

def showImage(name,image):
    img_show = cv2.resize(image, None, fx=0.2, fy=0.2)
    cv2.imshow(name,img_show)

def get_images(test=True):
    if test:
        above, below = util.get_test_images()
    else:
        for i in range(3):
            below = camera.take_picture(still_camera=True)

        mechanics_api.send_command("L", 1) #1 for turn off
        time.sleep(0.5)
        for i in range(3):
            above = camera.take_picture(still_camera=True)

    return above, below


def main():
    # Parse pieces
    above, below = get_images(test=False)
    above = util.relevant_section(above)
    below = util.relevant_section(below)
    showImage("above",above)
    showImage("below",below)
    cv2.waitKey(0)

    pieces = im_proc.get_pieces(above, below)
    print("pieces")

    #puzzle_obj = puzzle.Puzzle(pieces)

    # # Build puzzle
    # puzzle_obj.greedy()
    # puzzle_obj.display()

    # Execute puzzle


if __name__ == "__main__":
    main()

