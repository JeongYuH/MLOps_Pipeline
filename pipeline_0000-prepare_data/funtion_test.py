import prepare_data


if __name__=='__main__':
    caption_csv_path = '../data/test_data/caption/test_caption_data_drop_duple.csv'
    df, index_list = prepare_data.prepare_text(caption_csv_path)
    print(len(index_list))
