from pandas import *
from matplotlib.pyplot import * 	
from numpy import *	
from pymysql import *
from scipy.stats import kruskal, chisquare
from seaborn import *

## ~~~~~~ Connecting AWS to Python ~~~~~~ ##
host="[host here]"
port="[port here]"
dbname="[dbname here]"
user="[username here]"
password="[password here]"

mrdata = connect(host, user=user, port=port, passwd=password, db=dbname)
# [3]

## ~~~~~~ Data Cleaning Functions ~~~~~~ ##
# Replaces Periods in Attribute Names with Underscores.
def period_to_underscore(the_input):
	the_input.columns = [i.replace(".", "_") for i in the_input.columns]

	return the_input

# Transforms the Relevant Attributes into DateTime Types.
def datetime_converter(the_input, file):
	
	the_input['DateTime'] = to_datetime(the_input['DateTime'], format = "%m/%d/%Y %I:%M:%S %p")
	the_input['MonthYear'] = to_datetime(the_input['MonthYear'], format = "%m/%d/%Y %I:%M:%S %p")
	the_input['MonthYear'] = the_input.MonthYear.values.astype('datetime64[M]')

	if file == 1:
		the_input['Date_of_Birth'] = to_datetime(the_input['Date_of_Birth'], format = "%m/%d/%Y")
		the_input['Date_of_Birth'] = the_input['Date_of_Birth'].values.astype("datetime64[D]")

	return the_input

# Transforms the Age.upon.[x] Attributes into Numerics (with a Unit of Years).
def age_cleaning(the_input, file):
	if file == 1:
		week_cond = the_input.Age_upon_Outcome.str.contains("week+")
		the_input.loc[week_cond, "Age_upon_Outcome"] = to_numeric(the_input.Age_upon_Outcome[week_cond].replace("(week)+s*", "", regex = True))*0.01917811

		day_cond = the_input.Age_upon_Outcome.str.contains("day+", na = False)
		the_input.loc[day_cond, "Age_upon_Outcome"] = to_numeric(the_input.Age_upon_Outcome[day_cond].replace("(day)+s*", "", regex = True))*0.00273973

		month_cond = the_input.Age_upon_Outcome.str.contains("month+", na = False)
		the_input.loc[month_cond, "Age_upon_Outcome"] = to_numeric(the_input.Age_upon_Outcome[month_cond].replace("(month)+s*", "", regex = True))*0.083333545491

		year_cond = the_input.Age_upon_Outcome.str.contains("year+", na = False)
		the_input.loc[year_cond, "Age_upon_Outcome"] = to_numeric(the_input.Age_upon_Outcome[year_cond].replace("(year)+s*", "", regex = True))

		the_input.Age_upon_Outcome = to_numeric(the_input.Age_upon_Outcome)

	else:
		week_cond = the_input.Age_upon_Intake.str.contains("week+")
		the_input.loc[week_cond, "Age_upon_Intake"] = to_numeric(the_input.Age_upon_Intake[week_cond].replace("(week)+s*", "", regex = True))*0.01917811

		day_cond = the_input.Age_upon_Intake.str.contains("day+", na = False)
		the_input.loc[day_cond, "Age_upon_Intake"] = to_numeric(the_input.Age_upon_Intake[day_cond].replace("(day)+s*", "", regex = True))*0.00273973

		month_cond = the_input.Age_upon_Intake.str.contains("month+", na = False)
		the_input.loc[month_cond, "Age_upon_Intake"] = to_numeric(the_input.Age_upon_Intake[month_cond].replace("(month)+s*", "", regex = True))*0.083333545491

		year_cond = the_input.Age_upon_Intake.str.contains("year+", na = False)
		the_input.loc[year_cond, "Age_upon_Intake"] = to_numeric(the_input.Age_upon_Intake[year_cond].replace("(year)+s*", "", regex = True))

		the_input.Age_upon_Intake = to_numeric(the_input.Age_upon_Intake)

	return the_input

# Converts all <NAs> in the Sex.upon.[x] Attributes into "Unknown."
def unknown_sex(the_input, file):
	if file == 1:
		the_input.loc[the_input.Sex_upon_Outcome.isna(),"Sex_upon_Outcome"] = "Unknown"

	else:
		the_input.loc[the_input.Sex_upon_Intake.isna(), "Sex_upon_Intake"] = "Unknown"

	return the_input

# A Function that Runs the Previous Four Functions.
def total_clean(the_input, file):
	the_input = period_to_underscore(the_input)
	the_input = datetime_converter(the_input, file)
	the_input = age_cleaning(the_input, file)
	the_input = unknown_sex(the_input, file)

	return the_input

## ~~~~~~ Importing and Cleaning the Data ~~~~~~ ##
## When Importing Data, the following rows are removed:
### wherever Outcome.Type is <NA>.
### wherever Age.upon.Outcome is <NA>.
### Total Loss of Rows: 13.

# Data Import.
dog_outcomes = read_sql('select `Animal.ID`, DateTime, MonthYear, `Date.of.Birth`, `Outcome.Type`, `Outcome.Subtype`, `Sex.upon.Outcome`, `Age.upon.Outcome`, Breed, Color from raw_outcomes where `Animal.Type` = "Dog" and `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con=mrdata)
cat_outcomes = read_sql('select `Animal.ID`, DateTime, MonthYear, `Date.of.Birth`, `Outcome.Type`, `Outcome.Subtype`, `Sex.upon.Outcome`, `Age.upon.Outcome`, Breed, Color from raw_outcomes where `Animal.Type` = "Cat" and `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con=mrdata)
bird_outcomes = read_sql('select `Animal.ID`, DateTime, MonthYear, `Date.of.Birth`, `Outcome.Type`, `Outcome.Subtype`, `Sex.upon.Outcome`, `Age.upon.Outcome`, Breed, Color from raw_outcomes where `Animal.Type` = "Bird" and `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con=mrdata)
livestock_outcomes = read_sql('select `Animal.ID`, DateTime, MonthYear, `Date.of.Birth`, `Outcome.Type`, `Outcome.Subtype`, `Sex.upon.Outcome`, `Age.upon.Outcome`, Breed, Color from raw_outcomes where `Animal.Type` = "Livestock" and `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con=mrdata)
other_outcomes = read_sql('select `Animal.ID`, DateTime, MonthYear, `Date.of.Birth`, `Outcome.Type`, `Outcome.Subtype`, `Sex.upon.Outcome`, `Age.upon.Outcome`, Breed, Color from raw_outcomes where `Animal.Type` = "Other" and `Age.upon.Outcome` != "NA" and `Outcome.Type` != "NA";', con=mrdata)

# Cleaning the Data.
dog_outcomes = total_clean(dog_outcomes, 1)
cat_outcomes = total_clean(cat_outcomes, 1)
bird_outcomes = total_clean(bird_outcomes, 1)
livestock_outcomes = total_clean(livestock_outcomes, 1)
other_outcomes = total_clean(other_outcomes, 1)

## ~~~~~~ Initial Data Exploration ~~~~~~ ##
# Preparing the Data for Plotting.
## I am Plotting Frequency vs Outcome Types for All Animals.
outcome_types = DataFrame()
temp = DataFrame()

outcome_types['Outcome'] = dog_outcomes.loc[:,"Outcome_Type"]
outcome_types['Animal'] = "Dog"

temp['Outcome'] = cat_outcomes.loc[:,"Outcome_Type"]
temp['Animal'] = "Cat"

outcome_types = outcome_types.append(temp)

temp['Outcome'] = bird_outcomes.loc[:,"Outcome_Type"]
temp['Animal'] = "Bird"

outcome_types = outcome_types.append(temp)

temp['Outcome'] = livestock_outcomes.loc[:,"Outcome_Type"]
temp['Animal'] = "Livestock"

outcome_types = outcome_types.append(temp)

temp['Outcome'] = other_outcomes.loc[:,"Outcome_Type"]
temp['Animal'] = "Other"

outcome_types = outcome_types.append(temp)

# Plotting the Outcome Types.
countplot(x = "Outcome", hue = "Animal", data = outcome_types)
xlabel("Outcome Type")
ylabel("Frequency")
title("Frequency of Outcome Types for All Animals")
legend(loc = 1)
show()

### Works Cited
# [1] https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm

# [2] https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238

# [3] https://datascience-enthusiast.com/R/AWS_RDS_R_Python.html
