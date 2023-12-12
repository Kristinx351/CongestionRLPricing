python run_deltatoll.py --config_file "dataset/manual_16x3/config_undir.json" --thread 8 --updates_per_step 5 --date "3-18"
python run_formula.py --config_file "dataset/manual_16x3/config_undir.json" --thread 8 --updates_per_step 5 --date "3-19"
python run_nochange.py --config_file "dataset/manual_16x3/config_undir.json" --thread 8 --updates_per_step 5 --date "3-19"
python run_arbitrary.py --config_file "dataset/manual_16x3/config_undir.json" --thread 8 --updates_per_step 5 --date "3-19"
python run_OD.py --config_file "dataset/hangzhou_4x4/config.json" --thread 8 --updates_per_step 10 --date "5-7" --embed_dim 1 --compare_dim 16 --att_dim 4 