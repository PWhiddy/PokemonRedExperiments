cat saves_to_record.txt | xargs -I {} sh -c "python render_all_needed_grids.py {}"
