"""Functions for computing the odds of a pokemon"""


def calc_size_odds(size):
    """Calculate the odds of a specific size"""
    total = 0
    for i in range(129):
        for j in range(128):
            if size == i + j:
                total += 1
    return (129 * 128) / total


def calc_size_list_odds(sizes):
    """Calculate the odds of a list of height and weight"""
    total_odds = 0
    for height, weight in sizes:
        odds = calc_size_odds(height) * calc_size_odds(weight)
        total_odds += 1 / odds
    return 1 / total_odds


def calc_gender_odds(gender, gender_ratio):
    """Calculate the odds of a specific gender with a gender ratio"""
    if gender == 2 or gender_ratio in (255, 254, 0):
        return 1
    return 253 / sum(
        ((gender_val + 1) < gender_ratio) == gender for gender_val in range(253)
    )


def calc_expected_seeds(has_two_abilities, gender, gender_ratio, sizes):
    """Calculate the expected amount of seeds from an iv search"""
    return 2 ** 34 / (
        (2 if has_two_abilities else 1)
        * 25
        * calc_gender_odds(gender, gender_ratio)
        * calc_size_list_odds(sizes)
    )
