from lxml import etree


def test_merge(xml):
    tree_xml = etree.parse(xml)
    corpus = tree_xml.getroot()
    document_id_set = set()
    for document in corpus:
        document_id = document.attrib['id']
        if document_id in document_id_set:
            print("NOT OK")
        else:
            document_id_set.add(document_id)
    print("Number document: ", len(document_id_set))

    sentence_id_set = set()
    for document in corpus:
        for sentence in document:
            sentence_id = sentence.attrib['id']
            if sentence_id in sentence_id_set:
                print("NOT OK")
            else:
                sentence_id_set.add(sentence_id)
    print("Number sentence: ", len(sentence_id_set))


if __name__ == '__main__':
    raw_data = '../raw_data/BioInfer.xml'
    test_merge(raw_data)
