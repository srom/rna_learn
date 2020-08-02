import re


def parse_location(sequence_record):
    description = sequence_record.description.replace('\n', '').strip()

    m1 = re.match(r'^.*\[location=<?([0-9]+)..>?([0-9]+)\].*$', description)
    if m1 is not None:
        return [[int(m1[1]), int(m1[2])]], '+'

    m2 = re.match(r'^.*\[location=complement\(<?([0-9]+)..>?([0-9]+)\)\].*$', description)
    if m2 is not None:
        return [[int(m2[1]), int(m2[2])]], '-'

    m3 = re.match(r'^.*\[location=join\(([^\]]+)\)\].*$', description)
    if m3 is not None:
        joined_location = m3[1]
        try:
            locations = parse_joined_location(joined_location)
        except InvalidLocationError:
            raise InvalidLocationError(description)
            
        return locations, '+'

    m4 = re.match(r'^.*\[location=complement\(join\(([^\]]+)\)\)\].*$', description)
    if m4 is not None:
        joined_location = m4[1]
        try:
            locations = parse_joined_location(joined_location)
        except InvalidLocationError:
            raise InvalidLocationError(description)

        return locations, '-'

    raise InvalidLocationError(description)


def parse_joined_location(joined_location):
    locations = []
    for span_str in joined_location.split(','):
        span = span_str.strip()
        m = re.match(r'^<?([0-9]+)..>?([0-9]+)$', span)

        if m is not None:
            locations.append([int(m[1]), int(m[2])])
        else:
            raise InvalidLocationError(span)

    return locations


class InvalidLocationError(Exception):

    def __init__(self, message, *args, **kwargs):
        self.message = message
        super().__init__(*args, **kwargs)


def parse_chromosome_id(sequence_id):
    seq_id = sequence_id.strip()
    if seq_id.startswith('lcl|'):
        seq_id = seq_id[4:]

    parts = seq_id.split('_')

    return parts[0]
