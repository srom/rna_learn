import unittest

from Bio.SeqRecord import SeqRecord

from .sequence_utils import (
    parse_location, 
    InvalidLocationError,
    parse_chromosome_id,
    parse_sequence_type,
    InvalidSequenceTypeError,
)


DESCRIPTION_1 = """
lcl|AP014879.1_cds_BAV32329.1_3 [locus_tag=SCL_0003] 
[protein=DNA gyrase subunit B] [protein_id=BAV32329.1] 
[location=2853..5279] [gbkey=CDS]
"""

DESCRIPTION_2 = """
lcl|AP019551.1_cds_BBJ27171.1_38 [locus_tag=ATHSA_0039] 
[protein=NAD(FAD)-utilizing dehydrogenase] [protein_id=BBJ27171.1] 
[location=complement(37496..38770)] [gbkey=CDS]
"""

DESCRIPTION_3 = """
lcl|CP035901.1_cds_QHP94554.1_3203 [locus_tag=EXE55_16365] 
[protein=amino acid adenylation domain-containing protein] 
[protein_id=QHP94554.1] [location=join(2755259..2763738,2763740..2763750)] 
[gbkey=CDS]
"""

DESCRIPTION_4 = """
lcl|CP035901.1_cds_QHP94554.1_3203 [locus_tag=EXE55_16365] 
[protein=amino acid adenylation domain-containing protein] 
[protein_id=QHP94554.1] [gbkey=CDS]
[location=complement(join(2755259..2763738, 2763740..2763750))]
"""

DESCRIPTION_5 = """
lcl|AP014879.1_cds_BAV32329.1_3 [locus_tag=SCL_0003] 
[protein=DNA gyrase subunit B] [protein_id=BAV32329.1] 
[location=<2853..5279] [gbkey=CDS]
"""

DESCRIPTION_6 = """
lcl|CP035901.1_cds_QHP94554.1_3203 [locus_tag=EXE55_16365] 
[protein=amino acid adenylation domain-containing protein] 
[protein_id=QHP94554.1] [gbkey=CDS]
[location=complement(join(2755259..>2763738,2763740..2763750))]
"""

DESCRIPTION_INVALID = """
lcl|CP035901.1_cds_QHP94554.1_3203 [locus_tag=EXE55_16365] 
[protein=amino acid adenylation domain-containing protein] 
[protein_id=QHP94554.1] [location=complement(join(2755259..2763738,1))] 
[gbkey=CDS]
"""


class TestParseLocation(unittest.TestCase):

    def test_parse_location(self):
        with self.assertRaises(InvalidLocationError):
            record = SeqRecord('ATGC', description='')
            parse_location(record)

        with self.assertRaises(InvalidLocationError):
            record = SeqRecord('ATGC', description=DESCRIPTION_INVALID)
            parse_location(record)

        record = SeqRecord('ATGC', description=DESCRIPTION_1)
        location_list, strand = parse_location(record)
        self.assertEqual([[2853, 5279]], location_list)
        self.assertEqual('+', strand)

        record = SeqRecord('ATGC', description=DESCRIPTION_2)
        location_list, strand = parse_location(record)
        self.assertEqual([[37496, 38770]], location_list)
        self.assertEqual('-', strand)

        record = SeqRecord('ATGC', description=DESCRIPTION_3)
        location_list, strand = parse_location(record)
        self.assertEqual([[2755259, 2763738], [2763740, 2763750]], location_list)
        self.assertEqual('+', strand)

        record = SeqRecord('ATGC', description=DESCRIPTION_4)
        location_list, strand = parse_location(record)
        self.assertEqual([[2755259, 2763738], [2763740, 2763750]], location_list)
        self.assertEqual('-', strand)

        record = SeqRecord('ATGC', description=DESCRIPTION_5)
        location_list, strand = parse_location(record)
        self.assertEqual([[2853, 5279]], location_list)
        self.assertEqual('+', strand)

        record = SeqRecord('ATGC', description=DESCRIPTION_6)
        location_list, strand = parse_location(record)
        self.assertEqual([[2755259, 2763738], [2763740, 2763750]], location_list)
        self.assertEqual('-', strand)


class TestParseChromosomeId(unittest.TestCase):

    def test_parse_chromosome_id(self):
        chromosome_id = parse_chromosome_id('')
        self.assertEqual('', chromosome_id)

        chromosome_id = parse_chromosome_id('lcl|AP019551.1_cds_BBJ27171.1_38')
        self.assertEqual('AP019551.1', chromosome_id)

        chromosome_id = parse_chromosome_id('AP019551.1_cds_BBJ27171.1_38')
        self.assertEqual('AP019551.1', chromosome_id)


class TestParseSequenceType(unittest.TestCase):

    def test_parse_location(self):
        with self.assertRaises(InvalidSequenceTypeError):
            record = SeqRecord('ATGC', description='')
            parse_sequence_type(record)

        record = SeqRecord('ATGC', description=DESCRIPTION_1)
        sequence_type = parse_sequence_type(record)
        self.assertEqual('CDS', sequence_type)


if __name__ == '__main__':
    unittest.main()
