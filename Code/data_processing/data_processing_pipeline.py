import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import geoplot as gplt
import fire


# From Carolyn's Notebooks
def electoral_data_merge(precinct_data, block_data, precinct_shape, block_shape, dest_path):
    """
    Merges all the requisite data to run with the CS+ 2020 districting algorithm
    :param precinct_data: Filepath to precinct-level election results (CSV)
    :param block_data: Filepath to Census Data (CSV), block level (101 SUMLEV), 
    specifically over 18 population and subdivided by Hispanic/Latino status.
    :param precinct_shape: Filepath to shapefile for the state precincts
    :param block_shape: Filepath to census block shapefile
    :param dest_path: Destination path for the merged file
    """
    # Import all data files

    # 2016 Harvard US House of Representatives Precinct-Level Returns
    precinct_nat = pd.read_csv(precinct_data, dtype=str, encoding='ISO-8859-1')

    # 2016 NC precinct-level shapefiles: 
    # https://github.com/nvkelso/election-geodata/tree/master/data/37-north-carolina/statewide/2016
    precinct_geo = gpd.read_file(precinct_shape)

    # 2010 Census Summary File 1 population demographic data
    block_pop = pd.read_csv(block_data)

    # 2010 census block shapefiles:
    block_geo = gpd.read_file(block_shape)

    precinct_nc = precinct_nat[precinct_nat['state'] == 'North Carolina']
    precinct_nc.to_csv("Outputs/Data/2016_NC_Merged_Data/harvard_nc2016_2.csv")
    df = pd.read_csv("Outputs/Data/2016_NC_Merged_Data/harvard_nc2016_2.csv")
    df["loc_prec"] = df.jurisdiction + df.precinct # Create a common identifier
    summed = df.groupby(["jurisdiction", "county_fips", "loc_prec", "party"]).sum() # For each precinct, sum up all the votes for each party
    
    # If you want to keep the district number, just add district to the list above, but not necessary.
    summed.reset_index(inplace=True)
    one_hot = pd.get_dummies(summed["party"]) # One-hot encode the party variable (i.e. create dummy variables)
    summed = summed.join(one_hot)
    
    # Combines data so that each precinct is represented by one row, with variables that show # of dem, rep, lib votes
    summed['dem_votes'] = summed.votes * summed.democratic
    summed['rep_votes'] = summed.votes * summed.republican
    summed['lib_votes'] = summed.votes * summed.libertarian
    final = summed.groupby(["jurisdiction", "county_fips", "loc_prec"]).sum() 
    final.reset_index(inplace=True)

    precinct_geo['loc_prec'] = precinct_geo.COUNTY_NAM + precinct_geo.PREC_ID
    merged = precinct_geo.merge(final, on="loc_prec", how="outer")
    eday = merged[merged.PREC_ID.notna()] # Select ONLY the "actual" precincts, not the "absentee" rows
    eday.info()

    # Precinct's Proportion of <party> votes out of the entire county
    eday['dem_prop'] = eday['dem_votes']/(eday.groupby("jurisdiction")['dem_votes'].transform("sum"))
    eday['rep_prop'] = eday['rep_votes']/(eday.groupby("jurisdiction")['rep_votes'].transform("sum"))
    eday['lib_prop'] = eday['lib_votes']/(eday.groupby("jurisdiction")['lib_votes'].transform("sum"))

    # Create a new dataframe consisting ONLY of rows that represent absentee votes
    absentees = pd.DataFrame(columns=merged.columns)
    start_idx = merged.PREC_ID.isna().value_counts().loc[False]
    for i in range(start_idx, len(merged)):
        absentees = absentees.append(merged.iloc[i])

    # Group and sum absentee, provisional, etc. votes by precinct
    ga = absentees.groupby("jurisdiction").sum().loc[:, ["dem_votes", "rep_votes", "lib_votes"]]
    ga = ga.rename({'dem_votes': 'abs_dem', 'rep_votes': 'abs_rep', 'lib_votes': 'abs_lib'}, axis=1)

    precincts = eday.merge(ga, on="jurisdiction", how = "left")

    precincts.abs_dem.fillna(0, inplace=True)
    precincts.abs_rep.fillna(0, inplace=True)
    precincts.abs_lib.fillna(0, inplace=True)
    precincts.lib_prop.fillna(0, inplace=True)

    precincts["total_dem"] = precincts.dem_votes + (precincts.dem_prop * precincts.abs_dem)
    precincts["total_rep"] = precincts.rep_votes + (precincts.rep_prop * precincts.abs_rep)
    precincts["total_lib"] = precincts.lib_votes + (precincts.lib_prop * precincts.abs_lib)
    precincts["total_votes"] = precincts.total_dem + precincts.total_rep + precincts.total_lib

    # Write the neighboring precincts
    for index, row in precincts.iterrows():  
        neighbors = precincts[precincts.geometry.touches(row['geometry'])].loc_prec.tolist() 
        precincts.at[index, "my_neighbors"] = ", ".join(neighbors)
    
    # unique block identifier is the 'BLOCKID10' column for df and the 'GEOID' column for population data (pf)
    block_geo = block_geo.rename(columns={'BLOCKID10': 'GEOID'}) # rename the 'BLOCKID10' column to 'GEOID' for df
    block_geo['GEOID'] = block_geo.GEOID.astype('int64') # match data type of 'GEOID' column between datasets

    merged = block_geo.merge(block_pop, on='GEOID')

    merged['COUNTYFP10'] = merged.COUNTYFP10.astype('int64')
    merged['COUNTYFP10'] += 37000

    blocks = merged

     # County id processing for precinct data
    precincts.loc[precincts.county_fips.isna(), "county_fips"] = 37125 #EDITED COUNTY_FIP TO COUNTY_FIPS
    precincts.county_fips = precincts.county_fips.astype('int64') - 37000

    # Match coordinate reference systems
    blocks = blocks.to_crs(precincts.crs)

    # Use centroids for blocks
    blocks['geometry'] = blocks['geometry'].centroid

    # Merge blocks and precincts 
    merged = gpd.sjoin(blocks, precincts, how='right', op='within')

     # Rename census codes to make them more readable
    merged = merged.rename(columns={'POP10':'total_pop','P0110001': 'total_18+', 'P0110002': 'hispanic', 'P0110003': 'not_hispanic', 'P0110004': 'pop_1_race',
                'P0110005': 'white','P0110006': 'african_am',
            'P0110007': 'am_indian_native', 'P0110008': 'asian', 'P0110009':'hawaii/pac_is', 'P0110010':'other_race_alone', 'P0110011':'2+races'})


    # Drop some unnecessary/duplicated columns
    # EDITED COLUMN NAMES
    dataset = merged.drop(columns=["GEOID", "SUMLEV", "STATE", "COUNTY", "TRACT", "BLOCK", "PREC_ID", 
                               "ENR_DESC", "OF_PREC_ID", "COUNTY_ID", "jurisdiction", "county_fips", 
                               "year", "state_fips", "state_icpsr", "county_ansi", "county_lat", "county_long", "candidate_govtrack", 
                               "candidate_icpsr", "candidate_maplight", "democratic", "libertarian", "republican", "index_left"])

    # Extract precinct-level demographics from merged precinct-block data
    extract_demographics = (dataset.groupby(["loc_prec"]).sum()).loc[:,"HOUSING10":"2+races"]
    extract_demographics.reset_index()

    # Merge precinct-level demographic data with original precinct files (we don't need block data anymore)
    final = precincts.merge(extract_demographics, how="inner", on="loc_prec")
    final = final.drop(columns=["PREC_ID", 
                               "ENR_DESC", "OF_PREC_ID", "COUNTY_ID", "jurisdiction", "county_fips", 
                               "year", "state_fips", "state_icpsr", "county_ansi", "county_lat", "county_long", "candidate_govtrack", 
                               "candidate_icpsr", "candidate_maplight", "democratic", "libertarian", "republican"])

    # Add column with total minority voting age population
    final['minority'] = final['total_18+']-final['white']

    final.to_file(dest_path)

if __name__ == '__main__':
    fire.Fire({
        'electoral_data_merge' : electoral_data_merge
    })