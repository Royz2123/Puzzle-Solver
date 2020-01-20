# # import pygame module in this program
# import pygame
# import os
# import pygame
# from pygame.locals import *
#
#
# def load_image(name, colorkey=None):
#     fullname = os.path.join('data', name)
#     try:
#         image = pygame.image.load(fullname)
#     except pygame.error as message:
#         print('Cannot load image:', name)
#         raise SystemExit(message)
#     image = image.convert()
#     if colorkey is not None:
#         if colorkey is -1:
#             colorkey = image.get_at((0, 0))
#         image.set_colorkey(colorkey, RLEACCEL)
#     return image, image.get_rect()
#
#
# # # class PieceImage(pygame.sprite.Sprite):
# #     def __init__(self, piece, ):
# #         pygame.sprite.Sprite.__init__(self)
# #         self._chosen = False
# #
# #     def choose_piece(self):
# #         self._chosen = True
# #
# #     def update(self):
# #         """move the fist based on the mouse position"""
# #         if not self._chosen:
# #             return
# #
# #         pos = pygame.mouse.get_pos()
# #         self.rect.midtop = pos
# #         if self.punching:
# #             self.rect.move_ip(5, 10)
#
#
# def start_game(puzzle):
#     pygame.init()
#     pygame.mouse.set_visible(0)
#
#     # assigning values to X and Y variable
#     X = 1000
#     Y = 600
#     screen = pygame.display.set_mode((X, Y))
#     pygame.display.set_caption('Puzzle Solver')
#
#     pieces = puzzle.get_pieces()
#     images = [
#         pygame.image.load(r'.\\image_processing\\pieces\\game_images\\%s.png' % piece.get_name())
#         for piece in pieces
#     ]
#
#     # infinite loop
#     while True:
#         # completely fill the surface object
#         # with white colour
#         background = pygame.Surface(screen.get_size())
#         background = background.convert()
#         background.fill((200, 200, 200))
#
#         # Tile
#         font = pygame.font.Font(None, 36)
#         text = font.render("Puzzle Solver", 1, (10, 10, 10))
#         textpos = text.get_rect(centerx=background.get_width() / 2)
#         background.blit(text, textpos)
#
#         # copying the image surface object
#         # to the display surface object at
#         # (0, 0) coordinate.
#         screen.blit(images[0], (0, 0))
#         screen.blit(images[0], (100, 0))
#
#         # iterate over the list of Event objects
#         # that was returned by pygame.event.get() method.
#         for event in pygame.event.get():
#
#             # if event object type is QUIT
#             # then quitting the pygame
#             # and program both.
#             if event.type == pygame.QUIT:
#                 # deactivates the pygame library
#                 pygame.quit()
#
#                 # quit the program.
#                 quit()
#
#                 # Draws the surface object to the screen.
#
#
#
#             if event.type == pygame.MOUSEBUTTONDOWN:
#                 if event.button == 1:
#                     if rectangle.collidepoint(event.pos):
#                         rectangle_draging = True
#                         mouse_x, mouse_y = event.pos
#                         offset_x = rectangle.x - mouse_x
#                         offset_y = rectangle.y - mouse_y
#
#             elif event.type == pygame.MOUSEBUTTONUP:
#             if event.button == 1:
#                 rectangle_draging = False
#
#         elif event.type == pygame.MOUSEMOTION:
#         if rectangle_draging:
#             mouse_x, mouse_y = event.pos
#             rectangle.x = mouse_x + offset_x
#             rectangle.y = mouse_y + offset_y
#
