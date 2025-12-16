def test_get_data_paths_exist():
    app_path, credit_path = get_data_paths()
    assert app_path.exists()
    assert credit_path.exists()
