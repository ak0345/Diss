import pygame
import sys
import random
import math
from pygame.locals import *
from PIL import Image

# Initialize Pygame
pygame.init()
pygame.mixer.init()
mainClock = pygame.time.Clock()

pygame.mixer.music.load("background_music.mp3")
pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play(-1)

# Screen Dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), 0, 32)
pygame.display.set_caption('Mini-Catan')

# Fonts
FONT = pygame.font.SysFont(None, 40)
SMALL_FONT = pygame.font.SysFont(None, 30)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HIGHLIGHT = (244, 195, 44)  # Bright Gold
ROAD_COLOR = (186, 130, 47)  # Warm Earthy Brown
SETTLEMENT_COLOR = (197, 135, 32)  # Bronze-like color
BUTTON_COLOR = (207, 141, 2)  # Gold-brown
BUTTON_HOVER = (238, 166, 20)  # Brighter Gold
TEXT_COLOR = BLACK
INVENTORY_BG = (47, 34, 23)  # Dark Brown
BORDER_COLOR = BLACK
DROPDOWN_BG = (188, 158, 59)  # Muted Gold

# Hex Dimensions
HEX_SIZE = 80  
HEX_HEIGHT = 0.866 * HEX_SIZE

BOARD_CENTER_X = WINDOW_WIDTH // 2
BOARD_CENTER_Y = (WINDOW_HEIGHT // 2) + 50

# Game States
GAME_STATE = "START"

# AI Options
AI_TYPES = ["Random", "Weighted Random", "Reinforcement Learning"]

# Resource Colors
CLAY = (255, 0, 0)
WHEAT = (255, 215, 0)
SHEEP = (124, 252, 0)
WOOD = (0, 100, 0)
SAND = (139, 69, 19)

music_muted = False
dark_mode = False

#other vars
can_trade = True
trade_choice = True
trade_overlay_active = False

# Trading data placeholders
offer_input_boxes = []
request_input_boxes = []
trade_message = ""

# Game Variables
current_round = 1
current_player = 1
total_players = 4  # Adjust based on game settings
victory_points = 0
longest_road = ""

# Resource Inventory
resources = {
    "Clay": 0,
    "Wheat": 0,
    "Sheep": 0,
    "Wood": 0
}

# Structures Count
player_settlements = 0
total_settlements = 5  # Change as needed
player_roads = 0
total_roads = 10  # Change as needed

# Dice Roll
current_dice = 0

title_image = pygame.image.load("title.png")
title_image = pygame.transform.scale(title_image, (500, 180))
TITLE_X = (WINDOW_WIDTH - title_image.get_width()) // 2
TITLE_Y = 150  # Position above buttons

START_BUTTON_WIDTH = 500
START_BUTTON_HEIGHT = 60
START_BUTTON_X = (WINDOW_WIDTH - START_BUTTON_WIDTH) // 2
AI_DROPDOWN_WIDTH = 500
AI_DROPDOWN_HEIGHT = 60
AI_DROPDOWN_X = (WINDOW_WIDTH - AI_DROPDOWN_WIDTH) // 2

TRADE_OVERLAY_X = (WINDOW_WIDTH // 2) - 150  
TRADE_OVERLAY_Y = (WINDOW_HEIGHT // 2) - 150

INVENTORY_BOX_X = 20  
INVENTORY_BOX_Y = 20

ROLL_DICE_BUTTON_POS = (20, 310, 140, 50)
TRADE_BUTTON_POS = (WINDOW_WIDTH - 190, 650, 140, 50)
EXIT_BUTTON_POS = (WINDOW_WIDTH - 190, 20, 180, 50)
ACCEPT_TRADE_BUTTON_POS = (WINDOW_WIDTH // 2 - 180 - 90 - 60, 720, 200, 50)  
REJECT_TRADE_BUTTON_POS = (WINDOW_WIDTH // 2 - 90, 720, 200, 50)  
COUNTER_TRADE_BUTTON_POS = (WINDOW_WIDTH // 2 + 180 - 90 + 60, 720, 200, 50)
GAME_MUTE_BUTTON_POS = (WINDOW_WIDTH - 190, 90, 180, 50)
START_MUTE_BUTTON_POS = (WINDOW_WIDTH - 190, 20, 180, 50)
DARK_MODE_POS = (WINDOW_WIDTH - 190, 160, 180, 50)

# Adjusted Trade Input Boxes
TRADE_INPUT_X_START = TRADE_OVERLAY_X + 50  
TRADE_INPUT_Y_OFFER = TRADE_OVERLAY_Y + 80  
TRADE_INPUT_Y_REQUEST = TRADE_OVERLAY_Y + 140  
TRADE_INPUT_WIDTH = 60  
TRADE_INPUT_HEIGHT = 40  
TRADE_BUTTON_SEND_POS = (TRADE_OVERLAY_X + 70, TRADE_OVERLAY_Y + 200, 160, 50)
TRADE_CLOSE_BUTTON_POS = (TRADE_OVERLAY_X + 240, TRADE_OVERLAY_Y + 10, 100, 40)

class Board:
    """Represents the game board."""

    def __init__(self, size):
        self.hexes = []
        self.settlements = []
        self.roads = []
        self.size = size

        self.tiles = [CLAY, WHEAT] * 3 + [WHEAT, SHEEP, WOOD] * 4 + [SAND]
        random.shuffle(self.tiles)

        self.center_x = screen.get_rect().centerx
        self.center_y = screen.get_rect().centery

        self.create_hexes()
        self.create_settlements()
        self.create_roads()

    def create_hexes(self):
        """Creates hexagonal grid tiles."""
        self.hexes.append(Hex((self.center_x, self.center_y), self.size, self.tiles[0]))
        self.hexes.append(Hex((self.center_x, self.center_y - (2 * HEX_HEIGHT)), self.size, self.tiles[1]))
        self.hexes.append(Hex((self.center_x, self.center_y + (2 * HEX_HEIGHT)), self.size, self.tiles[2]))
        self.hexes.append(Hex((self.center_x - (1.5 * self.size), self.center_y - HEX_HEIGHT), self.size, self.tiles[3]))
        self.hexes.append(Hex((self.center_x + (1.5 * self.size), self.center_y - HEX_HEIGHT), self.size, self.tiles[4]))
        self.hexes.append(Hex((self.center_x - (1.5 * self.size), self.center_y + HEX_HEIGHT), self.size, self.tiles[5]))
        self.hexes.append(Hex((self.center_x + (1.5 * self.size), self.center_y + HEX_HEIGHT), self.size, self.tiles[6]))

    def create_settlements(self):
        """Generates settlement locations dynamically at hex corners."""
        settlement_positions = set()  # Use a set to store unique settlements

        for hexagon in self.hexes:
            for point in hexagon.points:
                # Ensure unique settlement positions
                rounded_point = (round(point[0]), round(point[1]))  # Avoid float precision errors
                settlement_positions.add(rounded_point)

        self.settlements = [Settlement((pos[0] - 8 // 2, pos[1] - 8 // 2)) for pos in settlement_positions]  # Convert to Settlement objects

    def create_roads(self):
        """Generates road locations dynamically at hex edges."""
        road_positions = set()  # Use a set to store unique roads

        for hexagon in self.hexes:
            for i in range(6):
                # Calculate the midpoint of the current edge
                p1 = hexagon.points[i]
                p2 = hexagon.points[(i + 1) % 6]
                midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

                # Round values to prevent floating point duplicates
                rounded_midpoint = (round(midpoint[0]), round(midpoint[1]))

                road_positions.add(rounded_midpoint)  # Add unique midpoint

        self.roads = [Road(pos) for pos in road_positions]  # Convert to Road objects

    def draw(self):
        """Draws the hexagons, settlements, and roads."""
        for hexagon in self.hexes:
            hexagon.draw()
        for road in self.roads:
            road.draw()
        for settlement in self.settlements:
            settlement.draw()

    def handle_mouse_hover(self, pos):
        """Handles hover effects on settlements and roads."""
        for settlement in self.settlements:
            settlement.check_hover(pos)
        for road in self.roads:
            road.check_hover(pos)

    def handle_mouse_click(self, pos):
        """Handles mouse clicks on settlements and roads."""
        for settlement in self.settlements:
            if settlement.check_click(pos):
                print(f"Settlement clicked at {settlement.position}")
        for road in self.roads:
            if road.check_click(pos):
                print(f"Road clicked at {road.position}")


class Settlement:
    """Represents a settlement."""

    def __init__(self, position):
        self.position = position
        self.color = SETTLEMENT_COLOR
        self.size = 16  # Square size
        self.valid = True

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.position[0], self.position[1], self.size, self.size))

    def check_hover(self, pos):
        x, y = pos
        if math.dist((x, y), self.position) < self.size and self.valid:
            self.color = HIGHLIGHT
        else:
            self.color = SETTLEMENT_COLOR

    def check_click(self, pos):
        if self.valid:
            return math.dist(pos, self.position) < self.size


class Road:
    """Represents a road."""
    
    def __init__(self, position):
        self.position = position  # Position of the road (midpoint of the edge)
        self.color = ROAD_COLOR   # Default road color
        self.size = 10            # Radius for hover and click detection
        self.valid = True         # Whether the road can be interacted with

    def draw(self):
        """Draws a small circle to represent the road."""
        pygame.draw.circle(screen, self.color, (int(self.position[0]), int(self.position[1])), self.size)

    def check_hover(self, pos):
        """Highlights the road when hovered over."""
        if self.valid and math.dist(pos, self.position) < self.size:
            self.color = HIGHLIGHT
        else:
            self.color = ROAD_COLOR

    def check_click(self, pos):
        """Checks if the road is clicked."""
        return self.valid and math.dist(pos, self.position) < self.size



class Hex:
    """Represents a hexagonal tile."""

    def __init__(self, center, size, color):
        self.center_x, self.center_y = center
        self.size = size
        self.color = color
        self.points = self.calculate_points()

    def calculate_points(self):
        """Calculates the corner points of the hexagon."""
        points = []
        for i in range(6):
            angle = math.radians(60 * i)
            points.append((self.center_x + self.size * math.cos(angle), self.center_y + self.size * math.sin(angle)))
        return points

    def get_edge_midpoint(self, edge_index):
        """Calculates the midpoint of a hex edge."""
        p1, p2 = self.points[edge_index], self.points[(edge_index + 1) % 6]
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def draw(self):
        pygame.draw.polygon(screen, self.color, self.points, 0)
        pygame.draw.polygon(screen, BLACK, self.points, 2)  # Outline

# Load and Process GIF
gif_path = r"C:\Users\foosh\OneDrive\Desktop\projects\DIss\mini_catan\catan.gif"
gif_image = Image.open(gif_path)
gif_frames = []
for frame in range(gif_image.n_frames):
    gif_image.seek(frame)
    frame_surface = pygame.image.fromstring(
        gif_image.tobytes(), gif_image.size, gif_image.mode
    )
    frame_surface = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
    gif_frames.append(frame_surface)
gif_frame_count = len(gif_frames)
gif_frame_duration = 100
gif_frame_index = 0
gif_frame_time = 0


class Button:
    """Represents a clickable button."""

    def __init__(self, text, rect, action=None):
        self.text = text
        self.rect = pygame.Rect(rect)
        self.action = action
        self.hovered = False

    def draw(self):
        color = BUTTON_HOVER if self.hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        text_surf = FONT.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def check_click(self, pos):
        if self.rect.collidepoint(pos) and self.action:
            self.action()


class Dropdown:
    """Represents a dropdown menu."""

    def __init__(self, options, rect):
        self.options = options
        self.rect = pygame.Rect(rect)
        self.expanded = False
        self.selected = options[0]
        self.hovered_option = None

    def draw(self):
        # Draw the main dropdown box
        pygame.draw.rect(screen, BUTTON_COLOR, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        text_surf = SMALL_FONT.render(self.selected, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)  # Center the text
        screen.blit(text_surf, text_rect)

        # Draw expanded options
        if self.expanded:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width, self.rect.height)
                color = BUTTON_HOVER if self.hovered_option == i else BUTTON_COLOR
                pygame.draw.rect(screen, color, option_rect)
                option_text_surf = SMALL_FONT.render(option, True, TEXT_COLOR)
                option_text_rect = option_text_surf.get_rect(center=option_rect.center)  # Center the text
                screen.blit(option_text_surf, option_text_rect)

    def check_hover(self, pos):
        if self.expanded:
            for i in range(len(self.options)):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width, self.rect.height)
                if option_rect.collidepoint(pos):
                    self.hovered_option = i
                    return
        self.hovered_option = None

    def check_click(self, pos):
        if self.rect.collidepoint(pos):
            self.expanded = not self.expanded
        elif self.expanded:
            for i in range(len(self.options)):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width, self.rect.height)
                if option_rect.collidepoint(pos):
                    self.selected = self.options[i]
                    self.expanded = False
                    return
            self.expanded = False
            

class InputBox:
    """Represents an input box for text input."""
    def __init__(self, rect, text=""):
        self.rect = pygame.Rect(rect)
        self.color = BUTTON_COLOR
        self.text = text
        self.font = SMALL_FONT
        self.active = False

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect, 2)
        text_surf = self.font.render(self.text, True, TEXT_COLOR)
        screen.blit(text_surf, (self.rect.x + 5, self.rect.y + 5))

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == KEYDOWN and self.active:
            if event.key == K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode
                

# Start Screen Functions
def start_game_pvc():
    global GAME_STATE
    GAME_STATE = "GAME"
    print(f"Starting Player vs Computer with AI type: {ai_dropdown.selected}")


def start_game_cvc():
    global GAME_STATE
    GAME_STATE = "GAME"
    print(f"Starting Computer vs Computer with AI type: {ai_dropdown.selected}")


# Start Screen Setup
start_buttons = [
    Button("Player vs Computer", (START_BUTTON_X, 450, START_BUTTON_WIDTH, START_BUTTON_HEIGHT), action=lambda: start_game_pvc()),
    Button("Computer vs Computer", (START_BUTTON_X, 550, START_BUTTON_WIDTH, START_BUTTON_HEIGHT), action=lambda: start_game_cvc()),
    Button("Mute", START_MUTE_BUTTON_POS, action=lambda: toggle_music()) #mute button in start
]
ai_dropdown = Dropdown(AI_TYPES, (AI_DROPDOWN_X, 350, AI_DROPDOWN_WIDTH, AI_DROPDOWN_HEIGHT))


def start_screen():
    """Displays the start screen with title image."""
    global gif_frame_index, gif_frame_time
    while GAME_STATE == "START":
        # Update GIF Frame
        current_time = pygame.time.get_ticks()
        if current_time - gif_frame_time > gif_frame_duration:
            gif_frame_index = (gif_frame_index + 1) % gif_frame_count
            gif_frame_time = current_time

        # Display Background Frame
        screen.blit(gif_frames[gif_frame_index], (0, 0))

        # Draw Title Image
        screen.blit(title_image, (TITLE_X, TITLE_Y))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION:
                pos = event.pos
                for button in start_buttons:
                    button.check_hover(pos)
                ai_dropdown.check_hover(pos)
            elif event.type == MOUSEBUTTONDOWN:
                pos = event.pos
                if ai_dropdown.expanded:
                    ai_dropdown.check_click(pos)
                else:
                    ai_dropdown.check_click(pos)
                    for button in start_buttons:
                        button.check_click(pos)

        # Draw Buttons and Dropdown
        for button in start_buttons:
            button.draw()
        ai_dropdown.draw()

        pygame.display.update()
        mainClock.tick(40)


def create_trade_overlay():
    """Creates the trade overlay with input boxes."""
    global offer_input_boxes, request_input_boxes
    offer_input_boxes = [InputBox((TRADE_INPUT_X_START + i * 70, TRADE_INPUT_Y_OFFER, TRADE_INPUT_WIDTH, TRADE_INPUT_HEIGHT)) for i in range(4)]
    request_input_boxes = [InputBox((TRADE_INPUT_X_START + i * 70, TRADE_INPUT_Y_REQUEST, TRADE_INPUT_WIDTH, TRADE_INPUT_HEIGHT)) for i in range(4)]


# Buttons for the trade and exit
roll_dice_button = Button("Roll Dice", ROLL_DICE_BUTTON_POS, action=lambda: roll_dice())
trade_button = Button("Trade", TRADE_BUTTON_POS, action=lambda: open_trade_overlay())
exit_button = Button("Exit", EXIT_BUTTON_POS, action=lambda: exit())
accept_trade_button = Button("Accept Trade", ACCEPT_TRADE_BUTTON_POS, action=lambda: accept_trade())
reject_trade_button = Button("Reject Trade", REJECT_TRADE_BUTTON_POS, action=lambda: reject_trade())
counter_trade_button = Button("Counter Trade", COUNTER_TRADE_BUTTON_POS, action=lambda: open_trade_overlay())
send_trade_button = Button("Send Trade", TRADE_BUTTON_SEND_POS, action=lambda: send_trade())
close_trade_button = Button("Close", TRADE_CLOSE_BUTTON_POS, action=lambda: close_trade_overlay())
game_mute_button = Button("Mute", GAME_MUTE_BUTTON_POS, action=lambda: toggle_music())
dark_mode_button = Button("Toggle Dark", DARK_MODE_POS, action=lambda: toggle_dark())

main_game_buttons = [trade_button, close_trade_button, reject_trade_button, 
                     exit_button, accept_trade_button, send_trade_button, 
                     counter_trade_button, game_mute_button, roll_dice_button, dark_mode_button
]
settlers = Board(HEX_SIZE)

def exit():
    global GAME_STATE, can_trade, trade_choice, trade_overlay_active
    global offer_input_boxes, request_input_boxes, trade_message
    global current_round, current_player, victory_points, current_dice, longest_road
    global player_settlements, player_roads, resources
    global settlers

    # Reset game state variables
    GAME_STATE = "START"
    can_trade = True
    trade_choice = False
    trade_overlay_active = False

    # Reset trading placeholders
    offer_input_boxes = []
    request_input_boxes = []
    trade_message = ""

    # Reset game variables
    current_round = 1
    current_player = 1
    victory_points = 0
    current_dice = 0
    longest_road = ""

    # Reset resource inventory
    resources = {
        "Clay": 0,
        "Wheat": 0,
        "Sheep": 0,
        "Wood": 0
    }

    # Reset player structures
    player_settlements = 0
    player_roads = 0

    # Reset the game board
    settlers = Board(HEX_SIZE)
    
def toggle_music():
    """Toggles music on/off."""
    global music_muted
    if music_muted:
        pygame.mixer.music.set_volume(0.5)  # Unmute (restore volume)
        music_muted = False
    else:
        pygame.mixer.music.set_volume(0)  # Mute
        music_muted = True
        
def toggle_dark():
    """Toggles dark mode on/off for main_game only."""
    global dark_mode, TEXT_COLOR, WHITE, BLACK, INVENTORY_BG, BUTTON_COLOR, BUTTON_HOVER, BORDER_COLOR, DROPDOWN_BG
    if dark_mode:
        # Light Mode
        WHITE, BLACK = (255, 255, 255), (0, 0, 0)
        TEXT_COLOR = BLACK
        INVENTORY_BG = (47, 34, 23)  # Dark Brown
        BUTTON_COLOR = (207, 141, 2)  # Gold-brown
        BUTTON_HOVER = (238, 166, 20)  # Brighter Gold
        BORDER_COLOR = BLACK
        DROPDOWN_BG = (188, 158, 59)  # Muted Gold
        dark_mode = False
    else:
        # Dark Mode
        WHITE, BLACK = (0, 0, 0), (255, 255, 255)
        TEXT_COLOR = WHITE
        INVENTORY_BG = (30, 22, 15)  # Darker background
        BUTTON_COLOR = (178, 123, 31)  # Darker gold
        BUTTON_HOVER = (237, 174, 19)  # Brighter golden brown
        BORDER_COLOR = WHITE
        DROPDOWN_BG = (42, 28, 7)  # Dark brown
        dark_mode = True


def open_trade_overlay():
    """Opens the trade overlay."""
    global trade_overlay_active
    trade_overlay_active = True
    create_trade_overlay()


def close_trade_overlay():
    """Closes the trade overlay."""
    global trade_overlay_active
    trade_overlay_active = False


def send_trade():
    """Handles sending the trade."""
    global trade_message
    offer = [int(box.text) if box.text.isdigit() else 0 for box in offer_input_boxes]
    request = [int(box.text) if box.text.isdigit() else 0 for box in request_input_boxes]

    trade_message = f"Offer: {offer}, Request: {request}"
    print(trade_message)
    close_trade_overlay()


def accept_trade():
    """Handles trade acceptance."""
    print("Trade accepted!")
    global trade_choice
    trade_choice = False


def reject_trade():
    """Handles trade rejection."""
    print("Trade rejected!")
    global trade_choice
    trade_choice = False
    
def roll_dice():
    """Simulates rolling two dice and updates the current dice value."""
    global current_dice
    current_dice = random.randint(1, 6) + random.randint(1, 6)
    print(f"Dice Rolled: {current_dice}")

def draw_inventory():
    """Draws the player's inventory and game state in the top-left corner."""
    
    # Draw Game Stats
    stats = [
        f"Round: {current_round}",
        f"VP: {victory_points}",
        f"Dice: {current_dice}",
        f"Settlements: {player_settlements}/{total_settlements}",
        f"Roads: {player_roads}/{total_roads}",
        f"Longest Road: Player"
    ]
    
    y_offset = INVENTORY_BOX_Y + 20
    for stat in stats:
        text_surface = SMALL_FONT.render(stat, True, BLACK)
        screen.blit(text_surface, (INVENTORY_BOX_X + 10, y_offset))
        y_offset += 30

    # Display Resources
    y_offset += 5
    for resource, amount in resources.items():
        text_surface = SMALL_FONT.render(f"{resource}: {amount}", True, BLACK)
        screen.blit(text_surface, (INVENTORY_BOX_X + 10, y_offset))
        y_offset += 30
    
    # Main Game Loop
def main_game():
    """Runs the main game loop."""
    global can_trade, trade_choice, trade_overlay_active
    create_trade_overlay()  # Initialize the trade overlay inputs

    while GAME_STATE == "GAME":
        screen.fill(WHITE)
        settlers.draw()
        
        # Draw Inventory
        draw_inventory()
        roll_dice_button.draw()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION:
                if not trade_overlay_active:  # Only check hover if no overlay
                    settlers.handle_mouse_hover(event.pos)
                for button in main_game_buttons:
                    button.check_hover(event.pos)
            elif event.type == MOUSEBUTTONDOWN:
                if trade_overlay_active:
                    close_trade_button.check_click(event.pos)
                else:
                    settlers.handle_mouse_click(event.pos)  # Only allow clicks when overlay is not active
                for button in main_game_buttons:
                    button.check_click(event.pos)

            if trade_overlay_active:
                for box in offer_input_boxes + request_input_boxes:
                    box.handle_event(event)

        # Draw trade button if can_trade is True
        if can_trade:
            trade_button.draw()
        # Draw trade choice buttons if trade_choice is True
        if trade_choice:
            accept_trade_button.draw()
            reject_trade_button.draw()
            counter_trade_button.draw()
        # Draw trade overlay if active
        if trade_overlay_active:
            pygame.draw.rect(screen, WHITE, (TRADE_OVERLAY_X, TRADE_OVERLAY_Y, 350, 350))  # Trade overlay background
            pygame.draw.rect(screen, BLACK, (TRADE_OVERLAY_X, TRADE_OVERLAY_Y, 350, 350), 2)  # Border
            for box in offer_input_boxes + request_input_boxes:
                box.draw()
            send_trade_button.draw()
            close_trade_button.draw()

        # Always draw the exit button
        exit_button.draw()
        game_mute_button.draw()
        dark_mode_button.draw()

        pygame.display.update()
        mainClock.tick(40)



# Run the Game
while True:
    if GAME_STATE == "START":
        start_screen()
    elif GAME_STATE == "GAME":
        main_game()
