from emulators.pyboy.memory_addresses import MemoryAddresses


class MemoryReader:
    def __init__(self, emulator):
        self.emulator = emulator
        self.ram = emulator.get_ram_state()
    
    def _bit_count(self,value):
        return value.bit_count()

    def read_battle_state(self):
        return self.ram[MemoryAddresses.BATTLE_STATE.value]
    
    def read_fist_pokemon_current_hp(self):
        return 256 * self.ram[MemoryAddresses.PLAYER_FIRST_POKEMON_ACTUAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_FIRST_POKEMON_ACTUAL_HP_2.value] #256 to correct 
    
    def read_second_pokemon_current_hp(self):
        return 256 * self.ram[MemoryAddresses.PLAYER_SECOND_POKEMON_ACTUAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_SECOND_POKEMON_ACTUAL_HP_2.value] #256 to correct 
    
    def read_third_pokemon_current_hp(self):
        return 256 * self.ram[MemoryAddresses.PLAYER_THIRD_POKEMON_ACTUAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_THIRD_POKEMON_ACTUAL_HP_2.value] #256 to correct 
    
    def read_fourth_pokemon_current_hp(self):
        ram = self.emulator.get_ram_state()
        return 256 * self.ram[MemoryAddresses.PLAYER_FOURTH_POKEMON_ACTUAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_FOURTH_POKEMON_ACTUAL_HP_2.value] #256 to correct 
    
    def read_fifth_pokemon_current_hp(self):
        ram = self.emulator.get_ram_state()
        return 256 * self.ram[MemoryAddresses.PLAYER_FIFTH_POKEMON_ACTUAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_FIFTH_POKEMON_ACTUAL_HP_2.value] #256 to correct 
    
    def read_sixth_pokemon_current_hp(self):
        ram = self.emulator.get_ram_state()
        return 256 * self.ram[MemoryAddresses.PLAYER_SIXTH_POKEMON_ACTUAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_SIXTH_POKEMON_ACTUAL_HP_2.value] #256 to correct 
    
    def read_fist_pokemon_max_hp(self):
        return 256 * self.ram[MemoryAddresses.PLAYER_FIRST_POKEMON_TOTAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_FIRST_POKEMON_TOTAL_HP_2.value] #256 to correct 
    
    def read_second_pokemon_max_hp(self):
        return 256 * self.ram[MemoryAddresses.PLAYER_SECOND_POKEMON_TOTAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_SECOND_POKEMON_TOTAL_HP_2.value] #256 to correct 
    
    def read_third_pokemon_max_hp(self):
        return 256 * self.ram[MemoryAddresses.PLAYER_THIRD_POKEMON_TOTAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_THIRD_POKEMON_TOTAL_HP_2.value] #256 to correct 
    
    def read_fourth_pokemon_max_hp(self):
        return 256 * self.ram[MemoryAddresses.PLAYER_FOURTH_POKEMON_TOTAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_FOURTH_POKEMON_TOTAL_HP_2.value] #256 to correct 
    
    def read_fifth_pokemon_max_hp(self):
        return 256 * self.ram[MemoryAddresses.PLAYER_FIFTH_POKEMON_TOTAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_FIFTH_POKEMON_TOTAL_HP_2.value] #256 to correct 
    
    def read_sixth_pokemon_max_hp(self):
        return 256 * self.ram[MemoryAddresses.PLAYER_SIXTH_POKEMON_TOTAL_HP_1.value] + self.ram[MemoryAddresses.PLAYER_SIXTH_POKEMON_TOTAL_HP_2.value] #256 to correct 
    

    def read_player_first_pokemon_level(self):
        return self.ram[MemoryAddresses.PLAYER_FIRST_POKEMON_LEVEL.value]

    def read_player_second_pokemon_level(self):
        return self.ram[MemoryAddresses.PLAYER_SECOND_POKEMON_LEVEL.value]

    def read_player_third_pokemon_level(self):
        return self.ram[MemoryAddresses.PLAYER_THIRD_POKEMON_LEVEL.value]

    def read_player_fourth_pokemon_level(self):
        return self.ram[MemoryAddresses.PLAYER_FOURTH_POKEMON_LEVEL.value]

    def read_player_fifth_pokemon_level(self):
        return self.ram[MemoryAddresses.PLAYER_FIFTH_POKEMON_LEVEL.value]

    def read_player_sixth_pokemon_level(self):
        return self.ram[MemoryAddresses.PLAYER_SIXTH_POKEMON_LEVEL.value]
    

    def read_opponent_first_pokemon_level(self):
        return self.ram[MemoryAddresses.OPPONENT_FIRST_POKEMON_LEVEL.value]

    def read_opponent_second_pokemon_level(self):
        return self.ram[MemoryAddresses.OPPONENT_SECOND_POKEMON_LEVEL.value]

    def read_opponent_third_pokemon_level(self):
        return self.ram[MemoryAddresses.OPPONENT_THIRD_POKEMON_LEVEL.value]

    def read_opponent_fourth_pokemon_level(self):
        return self.ram[MemoryAddresses.OPPONENT_FOURTH_POKEMON_LEVEL.value]

    def read_opponent_fifth_pokemon_level(self):
        return self.ram[MemoryAddresses.OPPONENT_FIFTH_POKEMON_LEVEL.value]

    def read_opponent_sixth_pokemon_level(self):
        return self.ram[MemoryAddresses.OPPONENT_SIXTH_POKEMON_LEVEL.value]
    
    def read_progress_map(self):
        return self.ram[MemoryAddresses.CURRENT_MAP_NUMBER.value]

    def read_player_current_position(self):
        return (self.ram[MemoryAddresses.CURRENT_PLAYER_POSITION_X.value],self.ram[MemoryAddresses.CURRENT_PLAYER_POSITION_Y.value])
    
    def read_pokemon_in_party(self):
        return self.ram[MemoryAddresses.POKEMON_IN_PARTY.value]
    
    def read_player_first_pokemon_name(self):
        return self.ram[MemoryAddresses.PLAYER_FIRST_POKEMON_NAME.value]
    def read_player_second_pokemon_name(self):
        return self.ram[MemoryAddresses.PLAYER_SECOND_POKEMON_NAME.value]

    def read_player_third_pokemon_name(self):
        return self.ram[MemoryAddresses.PLAYER_THIRD_POKEMON_NAME.value]

    def read_player_fourth_pokemon_name(self):
        return self.ram[MemoryAddresses.PLAYER_FOURTH_POKEMON_NAME.value]

    def read_player_fifth_pokemon_name(self):
        return self.ram[MemoryAddresses.PLAYER_FIFTH_POKEMON_NAME.value]

    def read_player_sixth_pokemon_name(self):
        return self.ram[MemoryAddresses.PLAYER_SIXTH_POKEMON_NAME.value]
    
    def read_bagdes_in_possesion(self):
        return self._bit_count(self.ram[MemoryAddresses.BADGES_IN_POSSESION.value])
    

    ##EVENTS
    def read_events_done(self):
        events_done = 0
        for i in range(MemoryAddresses.EVENT_FLAGS_START.value,MemoryAddresses.EVENT_FLAGS_END.value):
            events_done += self._bit_count(self.ram[i])
        return events_done

    def read_event_bits(self):
        return [
            (self.ram[i] >> (7 - j)) & 1  # Extrae cada bit directamente con shift y AND
            for i in range(MemoryAddresses.EVENT_FLAGS_START.value, MemoryAddresses.EVENT_FLAGS_END.value)
            for j in range(8)
        ]

    #----------------------------------Helpful functions----------------------------------
    def is_in_battle(self):
        return self.read_battle_state() != 0
    
    def get_sum_all_current_hp(self):
        return self.read_fist_pokemon_current_hp() + self.read_second_pokemon_current_hp() + self.read_third_pokemon_current_hp() + self.read_fourth_pokemon_current_hp() + self.read_fifth_pokemon_current_hp() + self.read_sixth_pokemon_current_hp()
    
    def get_sum_all_max_hp(self):
        return self.read_fist_pokemon_max_hp() + self.read_second_pokemon_max_hp() + self.read_third_pokemon_max_hp() + self.read_fourth_pokemon_max_hp() + self.read_fifth_pokemon_max_hp() + self.read_sixth_pokemon_max_hp()
    
    def get_sum_all_player_pokemon_level(self):
        return self.read_player_first_pokemon_level() + self.read_player_second_pokemon_level() + self.read_player_third_pokemon_level() + self.read_player_fourth_pokemon_level() + self.read_player_fifth_pokemon_level() + self.read_player_sixth_pokemon_level()
    
    def get_all_player_pokemon_level(self):
        return [self.read_player_first_pokemon_level(), self.read_player_second_pokemon_level(), self.read_player_third_pokemon_level(), self.read_player_fourth_pokemon_level(), self.read_player_fifth_pokemon_level(), self.read_player_sixth_pokemon_level()]
    
    def get_all_player_pokemon_name(self):
        return [self.read_player_first_pokemon_name(), self.read_player_second_pokemon_name(), self.read_player_third_pokemon_name(), self.read_player_fourth_pokemon_name(), self.read_player_fifth_pokemon_name(), self.read_player_sixth_pokemon_name()]
    
    def get_sum_all_opponent_pokemon_level(self):
        return self.read_opponent_first_pokemon_level() + self.read_opponent_second_pokemon_level() + self.read_opponent_third_pokemon_level() + self.read_opponent_fourth_pokemon_level() + self.read_opponent_fifth_pokemon_level() + self.read_opponent_sixth_pokemon_level()
    
    def get_sum_all_player_normalized_levels(self):
        return self.get_sum_all_player_pokemon_level() / 600 #100 because the maximun level in pokemon is 100
    
    def get_sum_all_opponent_normalized_levels(self):
        return self.get_sum_all_opponent_pokemon_level() / 600 #100 because the maximun level in pokemon is 100
    
    def get_difference_between_events(self):
        return ((MemoryAddresses.EVENT_FLAGS_END.value - MemoryAddresses.EVENT_FLAGS_START.value) * 8)
    
    def get_game_coords(self):
        x, y = self.read_player_current_position()
        map = self.read_progress_map()
        return (x,y,map)

    
