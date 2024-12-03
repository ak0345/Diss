from Board import Board
from enums import Biome, Resource, Structure, HexCompEnum

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
        case "cancel":
            return "cancel"
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
        case "cancel":
            return "cancel"
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
        case "cancel":
            return "cancel"
        case _:
            return None
        
def custom_input(prompt, p):
    user_input = input(prompt)
    
    # Check for the 'board' or 'inventory' commands
    if user_input == "board":
        print(board.get_board_array())  # Display the current board with hex numbers
        return custom_input(prompt, p)  # Recurse to get another input after showing the board
    elif user_input == "cheat":
        p.add_2_inv([10,10,10,10])
    elif user_input == "inventory":
        print(f"{p.name}'s Inventory: \n{p.inventory}")  # Show current player's inventory
        return custom_input(prompt, p)  # Recurse to get another input after showing the inventory
    elif user_input == "longest road":
        print(f"{board.longest_road_owner.name if board.longest_road_owner is not None else "No one"} is longest road owner")
        return custom_input(prompt, p)  # Recurse to get another input after showing the inventory
    elif user_input == "victory points":
        for p_ in board.players:
            print(f"{p_.name} has {p_.vp} Victory Points")
        return custom_input(prompt, p)
    elif user_input == "exit":
        print("Exitting and ending game...")
        exit()
    return user_input  # Otherwise return the normal input

def get_valid_input(prompt, validation_func, p):
    while True:
        user_input = custom_input(prompt, p).strip().lower()
        result = validation_func(user_input)
        if result is not None:
            return result
        else:
            print("Invalid input, please try again.")

def attempt_placement_until_success(board, player, prompt_hex, prompt_pos, structure_type, prev_coords=None):
    while True:
        hex_index = get_valid_input(prompt_hex, check_hn_name, player)
        if hex_index == "cancel":
            return None
        pos_obj = get_valid_input(prompt_pos, check_s_name if structure_type == Structure.ROAD else check_e_name, player)
        if pos_obj == "cancel":
            return None
        
        hex_obj = board.map_hexblocks[hex_index]
        # Try placing the structure and break if successful
        if board.turn_number > 0:
            if board.place_struct(player, hex_obj, pos_obj, structure_type):
                return (hex_index, pos_obj)
            else:
                print("Illegal placement, please try again.")
        else:
            if prev_coords and structure_type == Structure.ROAD:
                prev_set_coords = (board.map_hexblocks[prev_coords[0]], prev_coords[1])
                if hex_obj.check_road_next_to_settlement_in_first_turn(pos_obj, prev_set_coords):
                    if board.place_struct(player, hex_obj, pos_obj, structure_type):
                        return (hex_index, pos_obj)
                    else:
                        print("Illegal placement, please try again.")
                else:
                        print("Road must be placed next to settlement 2")
            else:
                if board.place_struct(player, hex_obj, pos_obj, structure_type):
                    return (hex_index, pos_obj)
                else:
                    print("Illegal placement, please try again.")


def validate_positive_integer(user_input):
    try:
        value = int(user_input)
        if value >= 0:
            return value
    except ValueError:
        if user_input == "cancel":
            return user_input
    return None

def validate_structure_type(user_input):
    if user_input in {"road", "settlement", "cancel"}:
        return user_input
    return None


game = True
win_vp = 5

p1_name = input("Player 1 Name: ").strip().lower()
p2_name = input("Player 2 Name: ").strip().lower()

board = Board([p1_name, p2_name])

board.make_board()

print("Hex Numbers: ", board.hex_nums())

print("Robber Location: ", board.robber_loc)

print("Hex Biomes: ", board.hex_biomes())

for p in board.players:
    p.add_2_inv([4,4,2,2])
    # Placement for the first settlement and road
    p.first_settlement = attempt_placement_until_success(board, p, f"Player {p.name} Place Your First Settlement ( Hex Num ): ", f"Player {p.name} Place Your First Settlement ( Edge Num ): ", Structure.SETTLEMENT)
    attempt_placement_until_success(board, p, f"Player {p.name} Place Your First Road ( Hex Num ): ", f"Player {p.name} Place Your First Road ( Side Num ): ", Structure.ROAD, p.first_settlement)
    #print(p.inventory)

for p in board.players:
    prev_coords = attempt_placement_until_success(board, p, f"Player {p.name} Place Your Second Settlement ( Hex Num ): ", f"Player {p.name} Place Your Second Settlement ( Edge Num ): ", Structure.SETTLEMENT)
    board.give_resources(p, 0, p.first_settlement)  # Resources are given only after the second settlement
    attempt_placement_until_success(board, p, f"Player {p.name} Place Your Second Road ( Hex Num ): ", f"Player {p.name} Place Your Second Road ( Side Num ): ", Structure.ROAD, prev_coords)
    #print(p.inventory)

while game:
    board.turn_number += 1
    print(f" Turn {board.turn_number}")
    for p in board.players:
        
        print("Resource Round")
        print(f"{p.name}'s Inventory: \n{p.inventory}")
        dice_val = board.roll_dice()
        print(f"Rolled dice with value {dice_val}")
        for p_ in board.players:
            board.give_resources(p_, dice_val)

        print("Action Round")
        print(f"{p.name}'s Inventory: \n{p.inventory}")

        # Ask for the first action (trade or build)
        while True:
            action_choice = custom_input("Would you like to Trade or Build? (Type 'Trade', 'Build', or 'End' to end your turn): ", p).strip().lower()
            
            if action_choice == "trade":
                # Trade Phase
                p_r = None
                trade_with = get_valid_input(f"Choose Player { [x.name for x in board.players] + ['Bank'] }: ", lambda x: x if x in [y.name for y in board.players if y.name != p.name]+['bank'] else None, p)
                for p2 in board.players + [board.bank]:
                    if p2.name == trade_with:
                        p_r = p2

                # Prompt for trade details
                break_flag = False
                while True:
                    print("Enter the cards you want to give: (cancel to cancel)")
                    w = get_valid_input("Wood (will be 2x with Bank)? ", validate_positive_integer, p)
                    if w == "cancel":
                        break_flag = True
                        break
                    b = get_valid_input("Brick (will be 2x with Bank)? ", validate_positive_integer, p)
                    if b == "cancel":
                        break_flag = True
                        break
                    s = get_valid_input("Sheep (will be 2x with Bank)? ", validate_positive_integer, p)
                    if s == "cancel":
                        break_flag = True
                        break
                    wh = get_valid_input("Wheat (will be 2x with Bank)? ", validate_positive_integer, p)
                    if wh == "cancel":
                        break_flag = True
                        break

                    if p_r == board.bank:
                        w, b, s, wh = w * 2, b * 2, s * 2, wh * 2

                    # Validate trade input
                    if all(give <= inv for give, inv in zip([w, b, s, wh], p.inventory)):
                        break
                    else:
                        print("Invalid input: You don't have enough cards. Please try again.")

                if not break_flag:
                    # Cards to receive
                    print("Enter the cards you want to receive: (cancel to cancel)")
                    w_r = get_valid_input("Wood? ", validate_positive_integer, p)
                    if w_r == "cancel":
                            break_flag = True
                    b_r = get_valid_input("Brick? ", validate_positive_integer, p)
                    if b_r == "cancel":
                            break_flag = True
                    s_r = get_valid_input("Sheep? ", validate_positive_integer, p)
                    if s_r == "cancel":
                            break_flag = True
                    wh_r = get_valid_input("Wheat? ", validate_positive_integer, p)
                    if wh_r == "cancel":
                            break_flag = True

                    # Confirm trade with player or bank
                    if p_r != board.bank and not break_flag:
                        while True:
                            accept_trade = custom_input(f"Does {trade_with} accept the trade? (y/n) ", p).strip().lower()
                            if accept_trade == "y":
                                if p.trade_I_with_p(p_r, [w, b, s, wh], [w_r, b_r, s_r, wh_r]):
                                    print("Trade successful!")
                                    break
                                else:
                                    print("Trade failed: Either party does not have the required resources.")
                            elif accept_trade == "n":
                                # Prompt for counteroffer
                                counter_check = custom_input(f"Send a counteroffer to {p.name}? (y/n) ", p).strip().lower()
                                if counter_check == "y":
                                    print("Enter the counteroffer:")
                                    while True:
                                        # Counteroffer details
                                        print("Cards to give:")
                                        w_c = get_valid_input("Wood? ", validate_positive_integer, p)
                                        if w_c == "cancel":
                                            break_flag = True
                                            break
                                        b_c = get_valid_input("Brick? ", validate_positive_integer, p)
                                        if b_c == "cancel":
                                            break_flag = True
                                            break
                                        s_c = get_valid_input("Sheep? ", validate_positive_integer, p)
                                        if s_c == "cancel":
                                            break_flag = True
                                            break
                                        wh_c = get_valid_input("Wheat? ", validate_positive_integer, p)
                                        if wh_c == "cancel":
                                            break_flag = True
                                            break

                                        if all(give <= inv for give, inv in zip([w_c, b_c, s_c, wh_c], p_r.inventory)):
                                            break
                                        else:
                                            print(f"{trade_with} doesn't have enough resources for this counteroffer. Try again.")

                                    if not break_flag:
                                        # Cards to receive in the counteroffer
                                        print("Cards to receive:")
                                        w_r_c = get_valid_input("Wood? ", validate_positive_integer, p)
                                        if w_r_c == "cancel":
                                                break_flag = True
                                        b_r_c = get_valid_input("Brick? ", validate_positive_integer, p)
                                        if b_r_c == "cancel":
                                                break_flag = True
                                        s_r_c = get_valid_input("Sheep? ", validate_positive_integer, p)
                                        if s_r_c == "cancel":
                                                break_flag = True
                                        wh_r_c = get_valid_input("Wheat? ", validate_positive_integer, p)
                                        if wh_r_c == "cancel":
                                                break_flag = True

                                    if not break_flag:
                                        # Present counteroffer to the original trader
                                        counter_accept = custom_input(f"Does {p.name} accept the counteroffer? (y/n) ", p).strip().lower()
                                        if counter_accept == "y":
                                            if p_r.trade_I_with_p(p, [w_c, b_c, s_c, wh_c], [w_r_c, b_r_c, s_r_c, wh_r_c]):
                                                print("Counteroffer accepted! Trade completed.")
                                                break
                                            else:
                                                print("Trade failed: Either party does not have the required resources.")
                                        elif counter_accept == "n":
                                            print(f"Trade between {p.name} and {trade_with} canceled.")
                                            break
                                        else:
                                            print("Invalid input, please respond with 'y' or 'n'.")
                                else:
                                    print("No counteroffer made. Trade canceled.")
                                    break
                            else:
                                print("Invalid input. Please enter 'y' or 'n'.")
                    else:
                        if p.trade_I_with_p(p_r, [w, b, s, wh], [w_r, b_r, s_r, wh_r]):
                            print("Trade with the bank completed successfully.")
                        else:
                            print("Trade failed: You do not have the required resources.")

                # Once trading is completed, return to the main loop for further actions
                print("Trade phase completed.")
            
            elif action_choice == "build":
                # Build Phase
                while True:
                    struct_str = custom_input("Which structure would you like to build? (Road/Settlement): ", p).strip().lower()
                    if struct_str == "road":
                        struct = Structure.ROAD
                    elif struct_str == "settlement":
                        struct = Structure.SETTLEMENT
                    elif struct_str == "cancel":
                        break
                    else:
                        print("Invalid input. Please choose either 'Road' or 'Settlement'.")
                        continue

                    # Attempt placement
                    cancelled = attempt_placement_until_success(
                        board, p, f"Player {p.name}, select Hex Pos: ", f"Player {p.name}, select Edge/Side Num: ", struct
                    )
                    if cancelled:
                        break

                    # Ask if they want to build another structure
                    another_build = custom_input("Would you like to build another structure? (y/n): ", p).strip().lower()
                    if another_build != "y":
                        break

            elif action_choice == "end":
                # End turn
                print("Ending turn.")
                break

            else:
                print("Invalid input. Please type 'Trade', 'Build', or 'End'.")

        if p.vp >= win_vp:
            print(f"Player {p.name} has won")
            game = False
            break