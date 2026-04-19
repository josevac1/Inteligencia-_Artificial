
    
values: list[int] = [8, 10.7, 17.1, 11.2, 13.5, 9.9, 14.9, 9.4, 9.4, 3.1, 12.7]

def count_peaks(values: list[int]):
    peaks: int = 0

    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            print(values[i])
            peaks += 1


    return peaks
    
print(count_peaks(values))
