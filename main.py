import image_processing.image_proc as im_proc
import image_processing.util as util
import image_processing.camera as camera
import time
import algorithmics.puzzle as puzzle
import cv2
import mechanics.mechanics_api as mechanics_api


TEST_MODE = False

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
        for i in range(5):
            below = camera.take_picture(still_camera=True)
        below = cv2.cvtColor(below, cv2.COLOR_BGR2GRAY)

        mechanics_api.send_command("L", 1) #1 for turn off
        time.sleep(0.5)
        for i in range(5):
            above = camera.take_picture(still_camera=True)

    above = util.relevant_section(above, test)
    below = util.relevant_section(below, test)
    return above, below


def main():
    # Parse pieces
    # above, below = get_images(test=TEST_MODE)
    #
    # showImage("above", above)
    # cv2.waitKey(0)
    # cv2.imwrite("Calibrate.jpg",above)
    # showImage("below", below)
    #
    # pieces = im_proc.get_pieces(above, below, TEST_MODE)
    # puzzle_obj = puzzle.Puzzle(pieces)
    #
    # # Build puzzle
    # puzzle_obj.greedy()
    # puzzle_obj.display()
    #
    # # Execute puzzle
    # command_list = puzzle_obj.create_command_list()
    command_list = [(946, 372, 1000, 0, 3.2484306699086156), (964, 1296,
    1000, 500, 0.07982998571223732), (425, 341, 1500, 0,
    3.5987887391976305), (948, 840, 1500, 500, 2.9222493989846066), (422,
    831, 2000, 0, 4.797228827480033), (463, 1322, 2000, 500,
    4.398851793654819)]

    # print(command_list)
    # mechanics_api.execute_command(command_list)

if __name__ == "__main__":
    main()

