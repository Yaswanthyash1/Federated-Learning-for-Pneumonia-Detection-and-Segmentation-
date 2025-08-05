rm ./verify_dataset.log

echo "Verifying dataset integrity..."
nnUNetv2_plan_and_preprocess -d 1 2 3 4 5 6 7 8 9 10 11 --verify_dataset_integrity > ./verify_dataset.log 2>&1
