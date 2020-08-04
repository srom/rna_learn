import re
import json


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


def parse_sequence_type(sequence_record):
    description = sequence_record.description.replace('\n', '').strip()

    m = re.match(r'^.*\[gbkey=([^\]]+)\].*$', description)

    if m is not None:
        return m[1].strip()
    
    raise InvalidSequenceTypeError(description)


class InvalidSequenceTypeError(Exception):

    def __init__(self, message, *args, **kwargs):
        self.message = message
        super().__init__(*args, **kwargs)


def parse_protein_information(record):
    protein_information = {}
    description = record.description.replace('\n', '').strip()

    m = re.match(r'^.*\[protein=([^\]]+)\].*$', description)
    if m is not None:
        protein_information['protein'] = m[1].strip()

    m = re.match(r'^.*\[protein_id=([^\]]+)\].*$', description)
    if m is not None:
        protein_information['protein_id'] = m[1].strip()

    m = re.match(r'^.*\[gene=([^\]]+)\].*$', description)
    if m is not None:
        protein_information['gene'] = m[1].strip()

    if len(protein_information) > 0:
        return protein_information
    else:
        return None


def get_location_range(start, end, strand, length):
    if strand == '+':
        return set(range(start, end + 1))
    else:
        start_ = (length - end) + 1
        end_ = (length - start) + 1
        return set(range(start_, end_ + 1))


def get_non_coding_records(chromosome_id, sequence, non_coding_ix):
    records = []
    current_sequence = []
    current_locations = []
    prev_loc = None
    for loc in non_coding_ix:
        sequence_index = loc - 1
        if prev_loc is None or prev_loc + 1 == loc:
            current_sequence.append(sequence[sequence_index])
            current_locations.append(loc)
        else:
            seq = ''.join(current_sequence)
            start = current_locations[0]
            end = current_locations[-1]
            location_json = json.dumps([[start, end]])
            records.append((chromosome_id, location_json, seq))

            current_sequence = [sequence[sequence_index]]
            current_locations = [loc]

        prev_loc = loc

    if len(current_sequence) > 0:
        seq = ''.join(current_sequence)
        start = current_locations[0]
        end = current_locations[-1]
        location_json = json.dumps([[start, end]])
        records.append((chromosome_id, location_json, seq))

    return records
