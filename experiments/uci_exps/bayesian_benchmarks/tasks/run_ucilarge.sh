echo "deleting results_large.db"
rm results_large.db

for split in {1..20}
#for split in {21..23}
do
    echo Split: $split

    # Elevators
    # Neural Linear
    python3 nl_regression.py --dataset wilson_elevators --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule

    # SGD
    python3 swag_regression.py --dataset wilson_elevators --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule

    # SWAG
    python3 swag_regression.py --dataset wilson_elevators --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr

    # ESS
    python3 swag_regression.py --dataset wilson_elevators --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr \
        --inference ess --temperature 20000
    # VI
    python3 swag_regression.py --dataset wilson_elevators --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr \
        --inference vi --temperature 15000 --prior_std=2

    # Protein
    # Neural Linear
    python3 nl_regression.py --dataset wilson_protein --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule 

    # SGD
    python3 swag_regression.py --dataset wilson_protein --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule

    # SWAG
    python3 swag_regression.py --dataset wilson_protein --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr

    # ESS
    python3 swag_regression.py --dataset wilson_protein --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr \
        --inference ess --temperature 10000 
    # VI
    python3 swag_regression.py --dataset wilson_protein --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr \
        --inference vi --temperature 10000

    # pol
    # Neural Linear
    python3 nl_regression.py --dataset wilson_pol --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule 

    # SGD
    python3 swag_regression.py --dataset wilson_pol --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule

    # SWAG
    python3 swag_regression.py --dataset wilson_pol --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr

    # ESS
    python3 swag_regression.py --dataset wilson_pol --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr  \
        --inference ess --temperature 5000 

    # VI
    python3 swag_regression.py --dataset wilson_pol --model_variance --dir test --split $split --epochs 200 --batch_size 200 \
        --lr_init 1e-3 --database_path results_large.db --noise_var --model_variance --no_schedule \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr \
        --inference vi --temperature 5000

    # keggd
    # Neural Linear
    python3 nl_regression.py --dataset wilson_keggdirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
        --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db

    # SGD
    python3 swag_regression.py --dataset wilson_keggdirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
        --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db

    # SWAG
    python3 swag_regression.py --dataset wilson_keggdirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
         --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db \
         --swag --swag_lr 1e-4 --subspace pca --swag_start 160 --double-bias-lr

    # ESS
    python3 swag_regression.py --dataset wilson_keggdirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
         --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db \
         --swag --swag_lr 1e-4 --subspace pca --swag_start 160 --double-bias-lr \
         --inference ess --temperature 10000
    # VI
    python3 swag_regression.py --dataset wilson_keggdirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
         --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db \
         --swag --swag_lr 1e-4 --subspace pca --swag_start 160 --double-bias-lr \
         --inference vi --temperature 10000

    # keggu
    # Neural Linear
    python nl_regression.py --dataset wilson_keggundirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
        --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db

    # SGD
    python swag_regression.py --dataset wilson_keggundirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
        --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db
    
    # SWAG
    python swag_regression.py --dataset wilson_keggundirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
        --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 160 --double-bias-lr 

    # ESS
    python swag_regression.py --dataset wilson_keggundirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
        --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 160 --double-bias-lr \
        --inference ess --temperature 10000
    
    # VI
	python swag_regression.py --dataset wilson_keggundirected --model_variance --dir test --split $split --epochs 200 --batch_size 400 \
     	--lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db \
     	--swag --swag_lr 1e-4 --subspace pca --swag_start 160 --double-bias-lr \
		--inference vi --temperature 10000

    # skillcraft
    # Neural Linear
    python3 nl_regression.py --dataset wilson_skillcraft --model_variance --dir test --split $split --epochs 100 --batch_size 400 \
        --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db

    # SGD
    python3 swag_regression.py --dataset wilson_skillcraft --model_variance --dir test --split $split --epochs 100 --batch_size 400 \
        --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db

    # SWAG    
    python3 swag_regression.py --dataset wilson_skillcraft --model_variance --dir test --split $split --epochs 100 --batch_size 400 \
        --lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db \
        --swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr

    # ESS
    python3 swag_regression.py --dataset wilson_skillcraft --model_variance --dir test --split $split --epochs 100 --batch_size 400 \
    	--lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db \
    	--swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr \
		--inference=ess --temperature=10000
		
    # VI
    python3 swag_regression.py --dataset wilson_skillcraft --model_variance --dir test --split $split --epochs 100 --batch_size 400 \
    	--lr_init 1e-3 --noise_var --model_variance --no_schedule --database_path results_large.db \
    	--swag --swag_lr 1e-4 --subspace pca --swag_start 80 --double-bias-lr \
		--inference=vi --temperature=10000
done
