import pygame
import sys
import random
import math
import time
import os
import importlib
import traceback
from pygame.locals import *
import numpy as np
import threading
import catan_agent

# Initialize Pygame
pygame.init()
pygame.mixer.init()
mainClock = pygame.time.Clock()

# Screen Dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), 0, 32)
pygame.display.set_caption('Mini-Catan')

# Fonts
FONT = pygame.font.SysFont(None, 40)
SMALL_FONT = pygame.font.SysFont(None, 30)
TINY_FONT = pygame.font.SysFont(None, 20)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
HIGHLIGHT = (244, 195, 44)  # Bright Gold
ROAD_COLOR = (186, 130, 47)  # Warm Earthy Brown
SETTLEMENT_COLOR = (197, 135, 32)  # Bronze-like color
BUTTON_COLOR = (207, 141, 2)  # Gold-brown
BUTTON_HOVER = (238, 166, 20)  # Brighter Gold
TEXT_COLOR = BLACK
INVENTORY_BG = (47, 34, 23)  # Dark Brown
BORDER_COLOR = BLACK
DROPDOWN_BG = (188, 158, 59)  # Muted Gold

# Resource Colors
CLAY = (185, 71, 0)    # Brick red
SAND = (255, 215, 0)   # Desert yellow
SHEEP = (124, 252, 0)  # Green
WOOD = (34, 100, 34)   # Forest green
WHEAT = (218, 165, 32) # Wheat gold
RESOURCE_COLORS = {
    'Clay': CLAY,
    'Wheat': WHEAT,
    'Sheep': SHEEP,
    'Wood': WOOD
}

# Game States
GAME_STATE = "START"

# AI Options - you can add more agents here
AI_TYPES = ["RandomAgent", "GreedyAgent"]

# Try to load background music
try:
    pygame.mixer.music.load("gui_assets/background_music.mp3")
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play(-1)
    music_available = True
except:
    print("Background music file not found, continuing without music")
    music_available = False

music_muted = False
dark_mode = False

# Hex Dimensions
HEX_SIZE = 80  
HEX_HEIGHT = 0.866 * HEX_SIZE

BOARD_CENTER_X = WINDOW_WIDTH // 2
BOARD_CENTER_Y = (WINDOW_HEIGHT // 2) + 50

# UI Elements dimensions
START_BUTTON_WIDTH = 500
START_BUTTON_HEIGHT = 60
START_BUTTON_X = (WINDOW_WIDTH - START_BUTTON_WIDTH) // 2
AI_DROPDOWN_WIDTH = 500
AI_DROPDOWN_HEIGHT = 60
AI_DROPDOWN_X = (WINDOW_WIDTH - AI_DROPDOWN_WIDTH) // 2

SPEED_SLIDER_WIDTH = 300
SPEED_SLIDER_HEIGHT = 30
SPEED_SLIDER_X = (WINDOW_WIDTH - SPEED_SLIDER_WIDTH) // 2
SPEED_SLIDER_Y = 650

# End Turn and Menu Buttons
END_ROUND_BUTTON_POS = (20, 700, 140, 50)
EXIT_BUTTON_POS = (WINDOW_WIDTH - 160, 20, 140, 50)
GAME_MUTE_BUTTON_POS = (WINDOW_WIDTH - 160, 90, 140, 50)
RESTART_BUTTON_POS = (WINDOW_WIDTH - 160, 160, 140, 50)

# Player colors for settlements and roads
PLAYER_COLORS_SETTLEMENT = {
    None: SETTLEMENT_COLOR,  # default color when unowned
    0: (0, 0, 255),         # Blue for player 0
    1: (255, 0, 0)          # Red for player 1
}

PLAYER_COLORS_ROAD = {
    None: ROAD_COLOR,        # default color when unowned
    0: (0, 0, 255),         # Blue for player 0
    1: (255, 0, 0)          # Red for player 1
}

# Game Variables
game = None
agent1 = None
agent2 = None
board = None
current_dice = 0
simulation_speed = 0.5  # Default speed (0-1 range, 0=fast, 1=slow)
is_running = True
simulation_thread = None
mini_catan_available = False
gym_available = False

# Check if required modules are available
try:
    import mini_catan
    import gymnasium as gym
    mini_catan_available = True
    gym_available = True
    print("mini_catan and gymnasium modules found")
except ImportError as e:
    print(f"Warning: {e}")
    print("Game will run in demo mode only (no actual game logic)")


class GuiBoard:
    """Represents the game board."""
    def __init__(self, size, colormap=None):
        self.hexes = []
        # Fixed positions for settlements and roads to match engine indices
        self.settlement_positions = [
            (436, 188), (476, 257), (436, 327), (356, 327),
            (316, 257), (356, 188), (316, 396), (236, 396),
            (196, 327), (236, 257), (556, 257), (596, 327),
            (556, 396), (476, 396), (436, 465), (356, 465),
            (316, 535), (236, 535), (196, 465), (596, 465),
            (556, 535), (476, 535), (436, 604), (356, 604)
        ]
        self.road_positions = [
            (460, 227), (460, 296), (400, 331), (340, 296),
            (340, 227), (400, 192), (340, 365), (280, 400),
            (220, 365), (220, 296), (280, 261), (580, 296),
            (580, 365), (520, 400), (460, 365), (520, 261),
            (460, 435), (400, 469), (340, 435), (340, 504),
            (280, 539), (220, 504), (220, 435), (580, 435),
            (580, 504), (520, 539), (460, 504), (460, 573),
            (400, 608), (340, 573)
        ]
        self.size = size
        self.tiles = colormap if colormap else [WOOD, CLAY, WHEAT, SHEEP, WOOD, CLAY, SAND]

        # Create the board elements
        self.center_x = screen.get_rect().centerx
        self.center_y = screen.get_rect().centery
        self.create_hexes()
        self.create_settlements()
        self.create_roads()

    def create_settlements(self):
        self.settlements = [Settlement(pos) for pos in self.settlement_positions]

    def create_roads(self):
        self.roads = [Road(pos) for pos in self.road_positions]

    def create_hexes(self):
        """Creates hexagonal grid tiles."""
        self.hexes.append(Hex((self.center_x, self.center_y - (2 * HEX_HEIGHT)), self.size, self.tiles[0])) # H1
        self.hexes.append(Hex((self.center_x - (1.5 * self.size), self.center_y - HEX_HEIGHT), self.size, self.tiles[1])) # H2
        self.hexes.append(Hex((self.center_x + (1.5 * self.size), self.center_y - HEX_HEIGHT), self.size, self.tiles[2])) # H3
        self.hexes.append(Hex((self.center_x, self.center_y), self.size, self.tiles[3])) # H4
        self.hexes.append(Hex((self.center_x - (1.5 * self.size), self.center_y + HEX_HEIGHT), self.size, self.tiles[4])) # H5
        self.hexes.append(Hex((self.center_x + (1.5 * self.size), self.center_y + HEX_HEIGHT), self.size, self.tiles[5])) # H6
        self.hexes.append(Hex((self.center_x, self.center_y + (2 * HEX_HEIGHT)), self.size, self.tiles[6])) # H7

    def draw(self):
        """Draws the hexagons, settlements, and roads."""
        for hexagon in self.hexes:
            hexagon.draw()
        for road in self.roads:
            road.draw()
        for settlement in self.settlements:
            settlement.draw()

    def update_from_engine(self):
        """
        Update the owner for each settlement and road based on engine state.
        """
        global game
        if game is None or not mini_catan_available:
            return

        try:
            # Retrieve current ownership from the engine
            settlement_owners = game.board.get_edges()  # length should be 24
            road_owners = game.board.get_sides()        # length should be 30

            # Update settlements (using the same ordering as settlement_positions)
            for i, settlement in enumerate(self.settlements):
                # Convert the engine value to our internal owner index:
                # (Assuming engine returns 0 for empty, and 1 for player 0, etc.)
                owner = settlement_owners[i]
                settlement.owner = None if owner == 0 else (owner - 1)

            # Update roads similarly
            for i, road in enumerate(self.roads):
                owner = road_owners[i]
                road.owner = None if owner == 0 else (owner - 1)
        except Exception as e:
            print(f"Error updating from engine: {e}")

    def handle_mouse_hover(self, pos):
        """Handles hover effects on settlements and roads."""
        for settlement in self.settlements:
            settlement.check_hover(pos)
        for road in self.roads:
            road.check_hover(pos)

    def handle_mouse_click(self, pos):
        """Handles mouse clicks on settlements and roads."""
        global game
        if game is None or not mini_catan_available:
            return

        try:
            for settlement in self.settlements:
                if settlement.check_click(pos):
                    idx = self._position_to_index(pos, 0)
                    if idx > -1:
                        if game.board.turn_number > 0:
                            game.step(1)  # 1 is for settlement
                        print(f"Settlement clicked at index {idx}")
                        game.step(idx)
                    
            for road in self.roads:
                if road.check_click(pos):
                    idx = self._position_to_index(pos, 1)
                    if idx > -1:
                        if game.board.turn_number > 0:
                            game.step(0)  # 0 is for road
                        print(f"Road clicked at index {idx}")
                        game.step(idx)
        except Exception as e:
            print(f"Error handling mouse click: {e}")

    def _position_to_index(self, pos, struct=None):
        """Converts (x, y) position to an index for the board placement."""
        positions = self.settlement_positions if struct < 1 else self.road_positions
        
        tolerance = 10  # Tolerance for click accuracy
        for index, (x, y) in enumerate(positions):
            if abs(pos[0] - x) <= tolerance and abs(pos[1] - y) <= tolerance:
                return index
        return -1


class Settlement:
    """Represents a settlement."""
    def __init__(self, position):
        self.position = position
        self.owner = None  # No owner by default
        self.size = 16  # Square size
        self.valid = True
        self.hover = False

    def draw(self):
        # Use the player color if owned; otherwise, the default settlement color
        color = PLAYER_COLORS_SETTLEMENT.get(self.owner, SETTLEMENT_COLOR)
        if self.hover and self.owner is None:
            color = HIGHLIGHT
        pygame.draw.rect(screen, color, (self.position[0] - self.size//2, self.position[1] - self.size//2, self.size, self.size))

    def check_hover(self, pos):
        x, y = pos
        self.hover = math.dist((x, y), self.position) < self.size and self.valid and self.owner is None

    def check_click(self, pos):
        if self.valid and self.owner is None:
            return math.dist(pos, self.position) < self.size
        return False


class Road:
    """Represents a road."""
    def __init__(self, position):
        self.position = position  # Midpoint of the edge
        self.owner = None  # No owner by default
        self.size = 10   # Radius for detection
        self.valid = True
        self.hover = False

    def draw(self):
        color = PLAYER_COLORS_ROAD.get(self.owner, ROAD_COLOR)
        if self.hover and self.owner is None:
            color = HIGHLIGHT
        pygame.draw.circle(screen, color, (int(self.position[0]), int(self.position[1])), self.size)

    def check_hover(self, pos):
        self.hover = math.dist(pos, self.position) < self.size and self.valid and self.owner is None

    def check_click(self, pos):
        return self.valid and self.owner is None and math.dist(pos, self.position) < self.size


class Hex:
    """Represents a hexagonal tile."""
    def __init__(self, center, size, color):
        self.center_x, self.center_y = center
        self.size = size
        self.color = color
        self.points = self.calculate_points()
        self.dice_value = random.randint(2, 12)  # Random dice value for visual purposes

    def calculate_points(self):
        """Calculates the corner points of the hexagon."""
        points = []
        for i in range(6):
            angle = math.radians(60 * i)
            points.append((self.center_x + self.size * math.cos(angle), 
                          self.center_y + self.size * math.sin(angle)))
        return points

    def draw(self):
        # Draw filled hexagon
        pygame.draw.polygon(screen, self.color, self.points, 0)
        pygame.draw.polygon(screen, BLACK, self.points, 2)  # Outline
        
        # Skip drawing numbers on desert (sand color)
        if self.color == SAND:
            return
            
        # Draw dice number in the center of the hex
        font = pygame.font.SysFont(None, 30)
        text = font.render(str(self.dice_value), True, BLACK)
        text_rect = text.get_rect(center=(self.center_x, self.center_y))
        
        # Draw a white circle behind the number for better visibility
        pygame.draw.circle(screen, WHITE, (self.center_x, self.center_y), 15)
        screen.blit(text, text_rect)


class Button:
    """Represents a clickable button."""
    def __init__(self, text, rect, action=None):
        self.text = text
        self.rect = pygame.Rect(rect)
        self.action = action
        self.hovered = False
        self.active = True

    def draw(self):
        if not self.active:
            color = GRAY
        else:
            color = BUTTON_HOVER if self.hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        text_surf = FONT.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos) and self.active

    def check_click(self, pos):
        if self.rect.collidepoint(pos) and self.active and self.action:
            self.action()
            return True
        return False

    def set_active(self, active):
        self.active = active


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
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

        # Draw expanded options
        if self.expanded:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, 
                                         self.rect.width, self.rect.height)
                color = BUTTON_HOVER if self.hovered_option == i else BUTTON_COLOR
                pygame.draw.rect(screen, color, option_rect)
                option_text_surf = SMALL_FONT.render(option, True, TEXT_COLOR)
                option_text_rect = option_text_surf.get_rect(center=option_rect.center)
                screen.blit(option_text_surf, option_text_rect)

    def check_hover(self, pos):
        if self.expanded:
            for i in range(len(self.options)):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, 
                                         self.rect.width, self.rect.height)
                if option_rect.collidepoint(pos):
                    self.hovered_option = i
                    return
        self.hovered_option = None

    def check_click(self, pos):
        if self.rect.collidepoint(pos):
            self.expanded = not self.expanded
            return True
        elif self.expanded:
            for i in range(len(self.options)):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, 
                                         self.rect.width, self.rect.height)
                if option_rect.collidepoint(pos):
                    self.selected = self.options[i]
                    self.expanded = False
                    return True
            self.expanded = False
        return False


class Slider:
    """Represents a slider control."""
    def __init__(self, rect, min_val=0, max_val=1, initial_val=0.5):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.dragging = False
        self.handle_radius = 12
        self.handle_pos = self.get_handle_pos()
        # Track if the mouse is over the handle or track
        self.hover_handle = False
        self.hover_track = False

    def get_handle_pos(self):
        return (self.rect.x + int((self.value - self.min_val) / 
                               (self.max_val - self.min_val) * self.rect.width), 
                self.rect.y + self.rect.height // 2)

    def draw(self):
        # Draw the slider track
        pygame.draw.rect(screen, BUTTON_COLOR, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        # Draw the handle
        self.handle_pos = self.get_handle_pos()
        handle_color = BUTTON_HOVER if self.hover_handle or self.dragging else WHITE
        pygame.draw.circle(screen, handle_color, self.handle_pos, self.handle_radius)
        pygame.draw.circle(screen, BLACK, self.handle_pos, self.handle_radius, 2)
        
        # Draw the value
        value_text = f"Speed: {int((1-self.value)*100)}%"
        text_surf = SMALL_FONT.render(value_text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(self.rect.centerx, self.rect.y - 20))
        screen.blit(text_surf, text_rect)

    def check_hover(self, pos):
        # Check if mouse is over the handle
        self.hover_handle = math.dist(pos, self.handle_pos) < self.handle_radius
        # Check if mouse is over the track (but not the handle)
        self.hover_track = (not self.hover_handle and 
                           self.rect.collidepoint(pos) and
                           abs(pos[1] - self.rect.centery) < 20)
        return self.hover_handle or self.hover_track

    def start_drag(self, pos):
        # Start dragging when clicking on handle OR track
        if self.hover_handle:
            self.dragging = True
            return True
        elif self.hover_track:
            # If clicking on track, immediately move handle to that position
            self.value = self.min_val + (pos[0] - self.rect.x) / self.rect.width * (self.max_val - self.min_val)
            self.value = max(self.min_val, min(self.max_val, self.value))
            self.dragging = True
            return True
        return False

    def update_drag(self, pos):
        if self.dragging:
            x = max(self.rect.x, min(pos[0], self.rect.right))
            self.value = self.min_val + (x - self.rect.x) / self.rect.width * (self.max_val - self.min_val)
            return True
        return False

    def end_drag(self):
        self.dragging = False


def convert_biome_to_color(biome_array):
    """Convert biome values to color values for display."""
    from mini_catan import enums
    # Default colormap for testing or when biome_array is not available
    default_colors = [WOOD, CLAY, WHEAT, SHEEP, WOOD, CLAY, SAND]
    
    # If no biome array or empty, return default colors
    if not biome_array:
        print("No biome array provided, using default colors")
        return default_colors
    
    colormap = []
    try:
        for b in biome_array:
            # Match the biome type to a color
            if b == enums.Biome.FOREST:  # FOREST
                colormap.append(WOOD)
            elif b == enums.Biome.HILLS:  # HILLS
                colormap.append(CLAY)
            elif b == enums.Biome.FIELDS:  # FIELDS
                colormap.append(WHEAT)
            elif b == enums.Biome.PASTURE:  # PASTURE
                colormap.append(SHEEP)
            elif b == enums.Biome.DESERT:  # DESERT
                colormap.append(SAND)
            else:
                colormap.append(BLACK)
    except Exception as e:
        print(f"Error converting biome to color: {e}")
        return default_colors  # Return default colors on error
            
    # If we somehow got fewer than 7 colors, add defaults to pad it out
    while len(colormap) < 7:
        colormap.append(default_colors[len(colormap) % len(default_colors)])
        
    return colormap


def create_agent(agent_name, player_index):
    """Create an agent instance based on agent name."""
    if not mini_catan_available:
        return None
        
    try:
        if agent_name.lower() == "none" or agent_name.lower() == "player" or agent_name.lower() == "human":
            return None
        
        # Try to import the agent module if it's not already imported
        if not agent_name.lower() in sys.modules:
            try:
                importlib.import_module(f"catan_agent.{agent_name}")
            except ImportError:
                print(f"Warning: Could not import module catan_agent.{agent_name}")
                return None
        
        # Create the agent
        agent = eval(f"catan_agent.{agent_name}.{agent_name}({player_index})")
        return agent
    except Exception as e:
        print(f"Error creating agent {agent_name}: {e}")
        return None


def start_game_pvc():
    """Start a new Player vs Computer game."""
    global GAME_STATE, game, board, agent2, agent1
    
    if not mini_catan_available or not gym_available:
        show_error_message("mini_catan or gymnasium module not available")
        return
    
    try:
        import mini_catan
        import gymnasium as gym
        
        GAME_STATE = "GAME"
        game = gym.make("MiniCatanEnv-v0")
        game = game.unwrapped
        game.reset()
        game.main_player = 0  # Player 0 is the human player
        
        # Create an agent for the computer (player 1)
        agent_type = ai1_dropdown.selected
        agent2 = create_agent(agent_type, 1)
        agent1 = None  # Human player
        
        if game.board and hasattr(game.board, 'get_hex_biomes'):
            colormap = convert_biome_to_color(game.board.get_hex_biomes())
            board = GuiBoard(HEX_SIZE, colormap)
        else:
            board = GuiBoard(HEX_SIZE)
            
        print(f"Starting Player vs Computer with AI type: {agent_type}")
        
    except Exception as e:
        print(f"Error starting game: {e}")
        traceback.print_exc()
        show_error_message(f"Error: {str(e)}")


def start_game_cvc():
    """Start a new Computer vs Computer game in a separate thread."""
    global GAME_STATE, game, board, agent1, agent2, simulation_thread, is_running
    
    if not mini_catan_available or not gym_available:
        show_error_message("mini_catan or gymnasium module not available")
        return
    
    try:
        import mini_catan
        import gymnasium as gym
        
        GAME_STATE = "GAME"
        game = gym.make("MiniCatanEnv-v0")
        game = game.unwrapped
        game.reset()
        
        # Create agents for both players
        agent1_type = ai1_dropdown.selected
        agent2_type = ai2_dropdown.selected
        agent1 = create_agent(agent1_type, 0)
        agent2 = create_agent(agent2_type, 1)
        
        if game.board and hasattr(game.board, 'get_hex_biomes'):
            colormap = convert_biome_to_color(game.board.get_hex_biomes())
            board = GuiBoard(HEX_SIZE, colormap)
        else:
            board = GuiBoard(HEX_SIZE)
            
        print(f"Starting Computer vs Computer with AI types: {agent1_type} vs {agent2_type}")
        
        # Start simulation in a separate thread
        is_running = True
        simulation_thread = threading.Thread(target=run_agent_simulation)
        simulation_thread.daemon = True  # Thread will close when main program exits
        simulation_thread.start()
        
    except Exception as e:
        print(f"Error starting game: {e}")
        traceback.print_exc()
        show_error_message(f"Error: {str(e)}")


def run_agent_simulation():
    """Run the agent vs agent simulation in a separate thread."""
    global game, agent1, agent2, simulation_speed, is_running
    
    if game is None or agent1 is None or agent2 is None:
        return
    
    obs, reward, done, trunc, info = None, None, None, None, None
    
    while is_running and not (hasattr(game, 'done') and game.done):
        try:
            # Determine current player
            if game.board.turn_number == 0 and hasattr(game, 'init_phase'):
                if game.init_phase < 4:
                    current_player_idx = game.init_phase // 2  # 0, 0, 1, 1
                else:
                    current_player_idx = 1 - ((game.init_phase - 4) // 2)  # 1, 1, 0, 0
            else:
                current_player_idx = game.current_player
            
            current_agent = agent1 if current_player_idx == 0 else agent2
            
            if obs is not None:
                move = current_agent.act(obs, game.board, game)
            else:
                move = current_agent.act(game.state, game.board, game)
            
            # Print move for debugging
            print(f"Agent {current_player_idx + 1} move: {move}")
            
            # Apply the move to the game
            obs, reward, done, trunc, info = game.step(move)
            
        except Exception as e:
            print(f"Error in agent simulation: {e}")
    
    print("Simulation thread ended")


def toggle_music():
    """Toggles music on/off."""
    global music_muted
    if not music_available:
        return
        
    if music_muted:
        pygame.mixer.music.set_volume(0.5)  # Unmute (restore volume)
        music_muted = False
    else:
        pygame.mixer.music.set_volume(0)  # Mute
        music_muted = True


def exit_game():
    """Exit to start screen."""
    global GAME_STATE, game, agent1, agent2, simulation_thread, is_running
    
    # Stop the simulation thread if it's running
    if simulation_thread is not None and simulation_thread.is_alive():
        is_running = False
        simulation_thread.join(timeout=1.0)
    
    # Reset game state
    GAME_STATE = "START"
    game = None
    agent1 = None
    agent2 = None


def end_turn():
    """End the current player's turn."""
    global game
    if game is None or (agent1 is not None and agent2 is not None):
        return  # Don't allow manual end turn in CvC mode
    
    try:
        print("Ending turn manually")
        game.step(4)  # 4 is the end turn action
    except Exception as e:
        print(f"Error ending turn: {e}")


def show_error_message(message):
    """Display an error message on the screen."""
    global GAME_STATE
    
    error_surface = pygame.Surface((600, 200))
    error_surface.fill(BLACK)
    error_surface.set_alpha(230)
    
    error_rect = error_surface.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
    screen.blit(error_surface, error_rect)
    
    lines = message.split('\n')
    y_offset = error_rect.y + 50
    
    for line in lines:
        text_surf = SMALL_FONT.render(line, True, WHITE)
        text_rect = text_surf.get_rect(center=(WINDOW_WIDTH//2, y_offset))
        screen.blit(text_surf, text_rect)
        y_offset += 30
    
    continue_text = SMALL_FONT.render("Click anywhere to continue", True, HIGHLIGHT)
    continue_rect = continue_text.get_rect(center=(WINDOW_WIDTH//2, error_rect.bottom - 40))
    screen.blit(continue_text, continue_rect)
    
    pygame.display.update()
    
    # Wait for user to click
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                waiting = False


def draw_inventory():
    """Draws the player's inventory and game state."""
    global game
    
    inventory_rect = pygame.Rect(20, 20, 200, 300)
    pygame.draw.rect(screen, INVENTORY_BG, inventory_rect)
    pygame.draw.rect(screen, BORDER_COLOR, inventory_rect, 2)
    
    if game is None or not mini_catan_available:
        # Draw demo info if no game is available
        y_offset = 30
        demo_info = [
            "Demo Mode",
            "mini_catan module",
            "not available",
            "",
            "This is a visual demo only",
            "No game logic active"
        ]
        
        for info in demo_info:
            text_surface = SMALL_FONT.render(info, True, WHITE)
            screen.blit(text_surface, (30, y_offset))
            y_offset += 30
        return
    
    try:
        current_player = game.current_player
        current_round = game.board.turn_number
        
        # Determine which player is the human (for PvC mode)
        human_player = 0 if agent1 is None else (1 if agent2 is None else None)
        
        # Get stats for display
        victory_points = [game.board.players[0].vp, game.board.players[1].vp]
        longest_road = game.board.get_longest_road_owner()
        
        # Draw game stats
        y_offset = 30
        stats = [
            f"Round: {current_round}",
            f"Current Player: {current_player + 1}",
            f"Dice: {game.dice_val if hasattr(game, 'dice_val') else 0}",
            f"Longest Road: {longest_road if longest_road > 0 else 'None'}",
            f"Blue VP: {victory_points[0]}",
            f"Red VP: {victory_points[1]}"
        ]
        
        # Add human player indicator if in PvC mode
        if human_player is not None:
            stats.insert(1, f"You are Player {human_player + 1} ({'Blue' if human_player == 0 else 'Red'})")
        
        for stat in stats:
            text_surface = SMALL_FONT.render(stat, True, WHITE)
            screen.blit(text_surface, (30, y_offset))
            y_offset += 30
        
        # Display resources for both players
        y_offset += 10
        for player_idx in range(2):
            player_color = "Blue" if player_idx == 0 else "Red"
            text_surface = SMALL_FONT.render(f"{player_color} Resources:", True, WHITE)
            screen.blit(text_surface, (30, y_offset))
            y_offset += 25
            
            inventory = game.board.players[player_idx].inventory if current_round > 0 else [0,0,0,0]
            resource_names = ['Wood', 'Clay', 'Wheat', 'Sheep']
            
            for i, (resource, amount) in enumerate(zip(resource_names, inventory)):
                color_rect = pygame.Rect(30, y_offset, 15, 15)
                pygame.draw.rect(screen, RESOURCE_COLORS.get(resource, WHITE), color_rect)
                pygame.draw.rect(screen, BLACK, color_rect, 1)
                
                text_surface = TINY_FONT.render(f"{resource}: {amount}", True, WHITE)
                screen.blit(text_surface, (50, y_offset))
                y_offset += 20
            
            y_offset += 10
    except Exception as e:
        # Display error in inventory area
        print(f"Error drawing inventory: {e}")
        text_surface = SMALL_FONT.render("Error drawing inventory", True, WHITE)
        screen.blit(text_surface, (30, 30))


# Load title image if available, otherwise create text-based title
title_image = None
try:
    title_image = pygame.image.load("gui_assets/title.png")
    title_image = pygame.transform.scale(title_image, (500, 180))
except:
    print("Title image not found, using text instead")

# Create UI elements
speed_slider = Slider((SPEED_SLIDER_X, SPEED_SLIDER_Y, SPEED_SLIDER_WIDTH, SPEED_SLIDER_HEIGHT), 
                     min_val=0.0, max_val=1.0, initial_val=0.5)

# Start Screen Setup
start_buttons = [
    Button("Player vs Computer", (START_BUTTON_X, 400, START_BUTTON_WIDTH, START_BUTTON_HEIGHT), action=lambda: start_game_pvc()),
    Button("Computer vs Computer", (START_BUTTON_X, 500, START_BUTTON_WIDTH, START_BUTTON_HEIGHT), action=lambda: start_game_cvc()),
    Button("Mute", (WINDOW_WIDTH - 160, 20, 140, 50), action=lambda: toggle_music()) #mute button in start
]

# AI dropdowns
ai1_dropdown = Dropdown(AI_TYPES, (AI_DROPDOWN_X, 300, AI_DROPDOWN_WIDTH, AI_DROPDOWN_HEIGHT))
ai2_dropdown = Dropdown(AI_TYPES, (AI_DROPDOWN_X, 600, AI_DROPDOWN_WIDTH, AI_DROPDOWN_HEIGHT))

# Game Buttons
end_turn_button = Button("End Turn", END_ROUND_BUTTON_POS, action=lambda: end_turn())
exit_button = Button("Exit", EXIT_BUTTON_POS, action=lambda: exit_game())
game_mute_button = Button("Mute", GAME_MUTE_BUTTON_POS, action=lambda: toggle_music())
restart_button = Button("Restart", RESTART_BUTTON_POS, action=lambda: exit_game())

# Group buttons for easier management
game_buttons = [end_turn_button, exit_button, game_mute_button, restart_button]

def main_game():
    """Runs the main game loop."""
    global game, agent1, agent2, simulation_speed, board
    
    if board is None:
        board = GuiBoard(HEX_SIZE)  # Create a default board if none exists
    
    while GAME_STATE == "GAME":
        screen.fill(WHITE)
        
        # Update and draw the board
        if board:
            try:
                board.update_from_engine()
                board.draw()
            except Exception as e:
                print(f"Error drawing board: {e}")
        
        # Draw inventory and game state
        draw_inventory()
        
        # Determine which player is human, if any
        human_player = None
        if agent1 is None:
            human_player = 0
        elif agent2 is None:
            human_player = 1
        
        # Check if it's game over
        game_over = False
        winner = None
        
        if game and mini_catan_available:
            try:
                if hasattr(game, 'done') and game.done:
                    game_over = True
                    # Determine winner based on victory points
                    vp0 = game.board.players[0].vp
                    vp1 = game.board.players[1].vp
                    if vp0 >= game.board.vp_to_win:
                        winner = 0
                    elif vp1 >= game.board.vp_to_win:
                        winner = 1
            except Exception as e:
                print(f"Error checking game state: {e}")
        
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION:
                pos = event.pos
                # Handle button hover
                for button in game_buttons:
                    button.check_hover(event.pos)
                # Only check board hover if human player and not game over
                if human_player is not None and not game_over and game and mini_catan_available:
                    if game.current_player == human_player:
                        board.handle_mouse_hover(event.pos)
                # Check slider hover
                if human_player is None:  # Only in CvC mode
                    speed_slider.check_hover(pos)
            elif event.type == MOUSEBUTTONDOWN:
                pos = event.pos
                # Handle button clicks
                button_clicked = False
                for button in game_buttons:
                    if button.check_click(event.pos):
                        button_clicked = True
                        break
                        
                # If no button was clicked, try other interactions
                if not button_clicked:
                    # Handle board clicks if human player and it's their turn
                    if human_player is not None and not game_over and game and mini_catan_available:
                        if game.current_player == human_player:
                            board.handle_mouse_click(event.pos)
                    # Handle slider drag start
                    if human_player is None:  # Only in CvC mode
                        speed_slider.start_drag(pos)
            elif event.type == MOUSEBUTTONUP:
                # End slider drag
                speed_slider.end_drag()
            elif event.type == MOUSEMOTION:
                # Update slider during drag
                if speed_slider.dragging:
                    if speed_slider.update_drag(event.pos):
                        simulation_speed = speed_slider.value
        
        # Handle AI moves in Player vs Computer mode
        if not game_over and human_player is not None and game and mini_catan_available:
            try:
                ai_player = 1 if human_player == 0 else 0
                current_agent = agent2 if human_player == 0 else agent1
                
                if game.current_player == ai_player and current_agent is not None:
                    move = current_agent.act(game.state, game.board, game)
                    print(f"AI move: {move}")
                    game.step(move)
            except Exception as e:
                print(f"AI error: {e}")
        
        # Draw game buttons
        for button in game_buttons:
            # Only show End Turn button if it's human player's turn
            if button == end_turn_button:
                if human_player is not None and game and mini_catan_available:
                    button.set_active(game.current_player == human_player)
                else:
                    button.set_active(False)
            button.draw()
        
        # Draw speed slider for CvC mode
        if human_player is None:  # Computer vs Computer mode
            speed_slider.draw()
        
        # Draw game over message if applicable
        if game_over and winner is not None:
            overlay = pygame.Surface((WINDOW_WIDTH, 150))
            overlay.set_alpha(200)
            overlay.fill(BLACK)
            screen.blit(overlay, (0, WINDOW_HEIGHT // 2 - 75))
            
            winner_text = f"Player {winner + 1} ({'Blue' if winner == 0 else 'Red'}) Wins!"
            text_surf = FONT.render(winner_text, True, WHITE)
            text_rect = text_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            screen.blit(text_surf, text_rect)
            
            restart_text = "Click 'Restart' to play again"
            restart_surf = SMALL_FONT.render(restart_text, True, WHITE)
            restart_rect = restart_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40))
            screen.blit(restart_surf, restart_rect)
            
        # Draw demo mode warning if needed
        if not mini_catan_available:
            demo_overlay = pygame.Surface((WINDOW_WIDTH, 80))
            demo_overlay.set_alpha(200)
            demo_overlay.fill(BLACK)
            screen.blit(demo_overlay, (0, WINDOW_HEIGHT - 80))
            
            demo_text = "DEMO MODE: mini_catan module not available. Visual demonstration only."
            demo_surf = SMALL_FONT.render(demo_text, True, (255, 100, 100))
            demo_rect = demo_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50))
            screen.blit(demo_surf, demo_rect)

        pygame.display.update()
        mainClock.tick(40)
def start_screen():
    """Displays the start screen with title image or text."""
    while GAME_STATE == "START":
        # Clear the screen with a background color
        screen.fill((50, 30, 10))  # Dark brown background
        
        # Draw title
        if title_image:
            title_x = (WINDOW_WIDTH - title_image.get_width()) // 2
            title_y = 100
            screen.blit(title_image, (title_x, title_y))
        else:
            # Draw text title if image not available
            title_text = FONT.render("MINI CATAN", True, HIGHLIGHT)
            title_rect = title_text.get_rect(center=(WINDOW_WIDTH // 2, 120))
            screen.blit(title_text, title_rect)
            
            subtitle_text = SMALL_FONT.render("A Simplified Settlers of Catan Implementation", True, WHITE)
            subtitle_rect = subtitle_text.get_rect(center=(WINDOW_WIDTH // 2, 160))
            screen.blit(subtitle_text, subtitle_rect)

        # Draw instructions
        instructions_text = SMALL_FONT.render("Select Game Mode and AI Types", True, WHITE)
        instructions_rect = instructions_text.get_rect(center=(WINDOW_WIDTH // 2, 250))
        screen.blit(instructions_text, instructions_rect)
        
        # Draw labels for dropdowns
        ai1_label = SMALL_FONT.render("Player 1 (Blue) AI:", True, WHITE)
        ai1_label_rect = ai1_label.get_rect(topleft=(AI_DROPDOWN_X, 270))
        screen.blit(ai1_label, ai1_label_rect)
        
        ai2_label = SMALL_FONT.render("Player 2 (Red) AI:", True, WHITE)
        ai2_label_rect = ai2_label.get_rect(topleft=(AI_DROPDOWN_X, 570))
        screen.blit(ai2_label, ai2_label_rect)
        
        # Show module availability status
        module_status = "Game Modules: "
        if mini_catan_available and gym_available:
            module_status += "Available"
            status_color = (0, 255, 0)  # Green
        else:
            module_status += "Not Available (Demo Mode Only)"
            status_color = (255, 0, 0)  # Red
            
        status_text = SMALL_FONT.render(module_status, True, status_color)
        status_rect = status_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50))
        screen.blit(status_text, status_rect)

        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION:
                pos = event.pos
                for button in start_buttons:
                    button.check_hover(pos)
                ai1_dropdown.check_hover(pos)
                ai2_dropdown.check_hover(pos)
            elif event.type == MOUSEBUTTONDOWN:
                pos = event.pos
                dropdowns_clicked = False
                
                # Handle dropdown clicks first
                if ai1_dropdown.expanded or ai2_dropdown.expanded:
                    dropdowns_clicked = ai1_dropdown.check_click(pos) or ai2_dropdown.check_click(pos)
                else:
                    dropdowns_clicked = ai1_dropdown.check_click(pos) or ai2_dropdown.check_click(pos)
                
                # Only handle button clicks if dropdowns weren't interacted with
                if not dropdowns_clicked:
                    for button in start_buttons:
                        button.check_click(pos)

        # Draw UI elements
        for button in start_buttons:
            button.draw()
        ai1_dropdown.draw()
        ai2_dropdown.draw()

        pygame.display.update()
        mainClock.tick(40)

# Main game loop
def run_game():
    """Main game loop that handles different game states."""
    global GAME_STATE
    
    try:
        # Display startup info
        print("Mini Catan GUI")
        print("==============")
        print(f"mini_catan module available: {mini_catan_available}")
        print(f"gymnasium module available: {gym_available}")
        if not mini_catan_available or not gym_available:
            print("Running in DEMO MODE - visual only, no game logic")
            
        while True:
            if GAME_STATE == "START":
                start_screen()
            elif GAME_STATE == "GAME":
                main_game()
    except KeyboardInterrupt:
        print("Game closed by user")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        # Clean shutdown
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    print("Welcome to Mini-Catan GUI")
    print("Initializing game...")
    run_game()