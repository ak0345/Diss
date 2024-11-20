from Board import Board
import random
from enums import Biome, Resource, Structure, HexCompEnum
#from Bank import Bank
from Hex import HexBlock
from Player import Player
from Die import Die

def check_hn_name(hn_str):
    match hn_str:
        case "h1":
            return 0
        case "h2":
            return 1
        case "h3":
            return 2
        case "h4":
            return 3
        case "h5":
            return 4
        case "h6":
            return 5
        case "h7":
            return 6
        case _:
            return None
        
def check_s_name(pos_str):
    match pos_str:
        case "s1":
            return HexCompEnum.S1
        case "s2":
            return HexCompEnum.S2
        case "s3":
            return HexCompEnum.S3
        case "s4":
            return HexCompEnum.S4
        case "s5":
            return HexCompEnum.S5
        case "s6":
            return HexCompEnum.S6
        case _:
            return None
        
def check_e_name(pos_str):
    match pos_str:
        case "e1":
            return HexCompEnum.E1
        case "e2":
            return HexCompEnum.E2
        case "e3":
            return HexCompEnum.E3
        case "e4":
            return HexCompEnum.E4
        case "e5":
            return HexCompEnum.E5
        case "e6":
            return HexCompEnum.E6
        case _:
            return None

def get_valid_input(prompt, validation_func):
    while True:
        user_input = input(prompt)
        result = validation_func(user_input)
        if result is not None:
            return result
        else:
            print("Invalid input, please try again.")

def attempt_placement_until_success(board, player, prompt_hex, prompt_pos, structure_type):
    while True:
        hex_index = get_valid_input(prompt_hex, check_hn_name)
        pos_obj = get_valid_input(prompt_pos, check_s_name if structure_type == Structure.ROAD else check_e_name)
        
        hex_obj = board.map_hexblocks[hex_index]
        # Try placing the structure and break if successful
        if board.place_struct(player, hex_obj, pos_obj, structure_type):
            return (hex_index, pos_obj)
        else:
            print("Illegal placement, please try again.")


game = True
win_vp = 3

p1_name = input("Player 1 Name: ")
p2_name = input("Player 2 Name: ")

board = Board([p1_name, p2_name], 1)

board.make_board()

print(board.hex_nums())

print(board.robber_loc)

print(board.hex_biomes())

for p in board.players:
    p.add_2_inv([4,4,2,2])
    # Placement for the first settlement and road
    p.first_settlement = attempt_placement_until_success(board, p, f"Player {p.name} Place Your First Settlement ( Hex Num ): ", f"Player {p.name} Place Your First Settlement ( Edge Num ): ", Structure.SETTLEMENT)
    attempt_placement_until_success(board, p, f"Player {p.name} Place Your First Road ( Hex Num ): ", f"Player {p.name} Place Your First Road ( Side Num ): ", Structure.ROAD)
    #print(p.inventory)

for p in board.players:
    attempt_placement_until_success(board, p, f"Player {p.name} Place Your Second Settlement ( Hex Num ): ", f"Player {p.name} Place Your Second Settlement ( Edge Num ): ", Structure.SETTLEMENT)
    board.give_resources(p, 0, p.first_settlement)  # Resources are given only after the second settlement
    attempt_placement_until_success(board, p, f"Player {p.name} Place Your Second Road ( Hex Num ): ", f"Player {p.name} Place Your Second Road ( Side Num ): ", Structure.ROAD)
    #print(p.inventory)


while game:
    board.turn_number += 1
    print(f" Turn {board.turn_number}")
    for p in board.players:
        print("Resource Round")
        print(f"{p.name}'s Inventory: \n{p.inventory}")
        dice_val = board.roll_dice()
        print(f"Rolled dice with value {dice_val}")
        board.give_resources(p, dice_val)

        print("Trade Round")
        print(f"{p.name}'s Inventory: \n{p.inventory}")
        check = input("Trade? (y/n) ")
        if check == 'y':
            p_r = None
            trade_with = input(f"Choose Player { [x.name for x in board.players]+["Bank"] }: ")
            for p2 in board.players+[board.bank]:
                if p2.name == trade_with:
                    p_r = p2

            #[Wood, Brick, Sheep, Wheat]
            while True:
                # Prompt for card amounts
                w = int(input("How many Wood cards (to give) (will be 2x with Bank)? "))
                b = int(input("How many Brick cards (to give) (will be 2x with Bank)? "))
                s = int(input("How many Sheep cards (to give) (will be 2x with Bank)? "))
                wh = int(input("How many Wheat cards (to give) (will be 2x with Bank)? "))

                if p_r == board.bank:
                    w = w*2
                    b = b*2
                    s = s*2
                    wh = wh*2

                # Check if the values are within the limits of the player's inventory
                if (w <= p.inventory[0] and 
                    b <= p.inventory[1] and 
                    s <= p.inventory[2] and 
                    wh <= p.inventory[3]):
                    break  # Exit the loop if all values are within limits
                else:
                    print("Invalid input: You do not have enough cards. Please enter values within your inventory limits.")

            w_r = int(input("How Much Wood Cards (to recieve)? "))
            b_r = int(input("How Much Brick Cards (to recieve)? "))
            s_r = int(input("How Much Sheep Cards (to recieve)? "))
            wh_r = int(input("How Much Wheat Cards (to recieve)? "))

            if p_r != board.bank:
                accept_trade = input(f"Does player {trade_with} accept the trade? (y/n) ")
            else:
                accept_trade = "y"

            if accept_trade == "y":
                p.trade_I_with_p(p_r, [w, b, s, wh], [w_r, b_r, s_r, wh_r])

        print("Build Round")
        print(f"{p.name}'s Inventory: \n{p.inventory}")
        build_check = None
        while build_check != "n":
            build_check = input("Build? (y/n) ")
            if build_check == "y":
                struct_str = input("Which Structure (Road or Settlement)? ")
                if struct_str == "Road":
                    struct = Structure.ROAD
                elif struct_str == "Settlement":
                    struct = Structure.SETTLEMENT
                else:
                    print("Invalid structure type. Please enter 'Road' or 'Settlement'.")
                    continue  # Ask again if the structure type is invalid
        
                attempt_placement_until_success(board, p, f"Player {p.name} Hex Pos: ", f"Player {p.name} Edge/Side Num: ", struct)
                
            elif build_check != "n":
                print("Invalid input. Please enter 'y' to build or 'n' to stop building.")

        if p.vp >= win_vp:
            print(f"Player {p.name} has won")
            game = False
            break