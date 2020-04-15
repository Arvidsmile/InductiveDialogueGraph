import glob 
from swda import Transcript
import pandas as pd 
from tqdm import tqdm

dataframe_list = []
counter = 0
desired_no_of_convos = 5000
foundLabels = set()
utterance_set = set()

tag_to_name = \
{
	'ft' : 'Thanking',
	'bk' : 'Response Acknowledgement',
	'qy^d' : 'Declarative Yes-No Question',
	'bf' : 'Summarize/reformulate',
	'^q' : 'Quotation',

	'qw' : 'Wh-Question',
	'ar' : 'Reject',
	'fc' : 'Conventional-Closing',
	'qy' : 'Yes-No-Question',
	'b^m' : 'Repeat-Phrase',
	
	'ba' : 'Appreciation',
	'^g' : 'Tag-Question',
	'sd' : 'Statement-Non-Opinion',
	'br' : 'Signal-Non-Understanding',
	'qo' : 'Open-Question',

	'^h' : 'Hold before answer/agreement',
	'na' : 'Affirmative non-yes answer',
	'oo_co_cc' : 'Offers, Options, Commits',
	'qw^d' : 'Declarative Wh-Question',
	'no' : 'Other answers',

	'x' : 'Non-verbal',
	'fp' : 'Conventional-opening',
	'b' : 'Acknowledge (Backchannel)',
	'arp_nd' : 'Dispreferred answers',
	'bh' : 'Backchannel in question form',

	'h' : 'Hedge',
	'nn' : 'No Answers',
	'%' : 'Uninterpretable',
	't1' : 'Self-talk',
	'fo_o_fw_"_by_bc' : 'Other',

	'aa' : 'Agree/Accept',
	'aap_am' : 'Maybe/Accept-part',
	'bd' : 'Downplayer',
	'fa' : 'Apology',
	'ny' : 'Yes answers',

	'ad' : 'Action-directive',
	'qh' : 'Rhetorical-Question',
	'qrr' : 'Or-Clause',
	'^2' : 'Collaborative Completion',
	'ng' : 'Negative non-no answers',

	'sv' : 'Statement-Opinion',
	't3' : '3rd-party-talk',
	'+' : 'Segment',
}



# Find all names of transcripts in swda corpus 
for i in tqdm(glob.glob("./swda/sw**utt/*")):
	# Store name ID of conversation
	ID = i[7:14] + '-' + i[15:27]

	# Read file and 
	# go through all utterances
	dialogue = Transcript(i, 'swda/swda-metadata.csv')

	for utter in dialogue.utterances:
		utter_DA = tag_to_name[utter.damsl_act_tag()] # <-- add dictionary cleanup
		foundLabels.add(utter_DA)
		utterance_text = ' '.join(utter.pos_words())
		utterance_set.add(utterance_text)
		actor = utter.caller
		dataframe_list.append({'Dialogue ID' : ID,
							'Actor' : actor,
							'Utterance' : utterance_text,
							'Dialogue Act' : utter_DA})
	counter += 1
	if counter == desired_no_of_convos:
		break

print(len(foundLabels))
print(len(utterance_set))

df = pd.DataFrame(dataframe_list)
# pd.to_pickle(df, f'SWDA_{counter}_dialogues.pickle')
df.to_csv(f'SWDA_{counter}_dialogues.csv')
for i in foundLabels:
	print(i)
