"""
Builds the 2026 News & Disinformation Benchmark Dataset for The Turing Project.
Generates balanced, diverse training and holdout test sets across real reporting,
satire, rural welfare scams, panic hoaxes, and medical misinformation.
"""

import json
import os

REAL_NEWS_SAMPLES = [
    # Wire & Global Journalism
    "According to Reuters, the Federal Reserve decided to maintain its benchmark interest rate following the conclusion of its two-day policy meeting in Washington.",
    "The Associated Press reported that international diplomatic envoys gathered in Geneva to negotiate a regional maritime transit treaty.",
    "BBC News confirms that European regulators have finalized updated safety standards for civilian uncrewed aerial vehicles operating in urban corridors.",
    "Official data released by the Bureau of Labor Statistics indicates that consumer price inflation moderated to 2.4 percent annualized.",
    "According to the World Health Organization, global surveillance data shows a thirty percent reduction in malaria transmission in certified elimination zones.",
    "A peer-reviewed clinical trial published in The Lancet confirms that the novel mRNA therapeutic demonstrated efficacy in reducing recurrence rates.",
    "Official statistics from the Ministry of Finance showed tax collections expanded by eleven percent year-over-year in the latest quarterly filing.",
    "The Agence France-Presse reported that archaeologists in southern Greece uncovered a bronze age administrative seal dating to the fourteenth century BCE.",
    "In a formal statement issued by the United Nations High Commissioner for Refugees, humanitarian convoys successfully reached three distribution hubs.",
    "NASA and the European Space Agency confirmed the successful orbital insertion of the joint atmospheric monitoring satellite.",
    "According to official figures from the Reserve Bank of India, foreign direct investment inflows reached a four-quarter high in manufacturing sectors.",
    "A study published in the journal Nature demonstrates that coral reef restoration projects with thermal-tolerant strains show forty percent higher survival.",
    "The Centers for Disease Control and Prevention released updated seasonal influenza guidance recommending vaccination prior to late autumn.",
    "Associated Press reports that municipal authorities in Tokyo have commenced construction on expanded underground stormwater containment cisterns.",
    "According to an official press release from the Department of Agriculture, grain reserves remain at adequate buffer stock levels for the winter harvest.",
    "Reuters reports that international maritime shipping rates stabilized after alternative canal transit schedules were coordinated by port operators.",
    "A report published in the Proceedings of the National Academy of Sciences analyzed fifty years of meteorological data from high-altitude stations.",
    "Government of India portal announced the formal schedule for civil service examinations across all designated regional testing centers.",
    "According to the BBC, engineers in Scotland completed installation of the largest floating offshore wind turbine array to date.",
    "The World Bank published its regional economic outlook, projecting moderate growth across emerging South Asian economies over the next fiscal cycle.",
    "Official statement released by the Ministry of Education outlines new STEM curriculum frameworks for secondary schools starting next academic session.",
    "According to Reuters, energy ministers concluded bilateral agreements on cross-border transmission grid interconnections.",
    "Researchers from Stanford University published findings on low-power semiconductor architectures in the IEEE Journal of Solid-State Circuits.",
    "The Associated Press reported that conservation agencies observed an increase in the nesting population of loggerhead sea turtles.",
    "Statement released by the International Energy Agency indicates global solar power generation capacity expanded by twenty-two percent in the past year.",
    "According to official court filings, the appellate bench reserved its judgment on statutory administrative interpretation after hearing concluding arguments.",
    "BBC World Service reported that the regional transport authority completed electrification of the main regional freight railway line.",
    "The Indian Meteorological Department issued a seasonal advisory forecasting normal monsoon rainfall distribution across northern agricultural belts.",
    "Reuters reports that municipal councils approved new zoning regulations aimed at increasing affordable housing units near suburban transit stations.",
    "Scientific researchers confirmed in a paper published in Science that ancient permafrost core samples revealed previously undocumented microbial enzymes.",
    "Official bulletin from the Ministry of External Affairs confirmed the bilateral trade delegation visit concluded with five signed cooperation memoranda.",
    "The Associated Press reported that university researchers developed a water filtration membrane utilizing recycled agricultural husks.",
    "Public health officials in Ontario announced expanded pediatric vaccination clinics across regional healthcare networks.",
    "Reuters wire dispatch reports that the sovereign debt rating was affirmed with a stable outlook following conservative fiscal deficit management.",
    "According to data from the National Oceanic and Atmospheric Administration, sea surface temperature anomalies eased during the late spring transition.",
    "The Supreme Court of India delivered a unanimous verdict upholding standard contractual arbitration procedures under the commercial code.",
    "Associated Press reports that the international botanical conservation congress adopted a ten-point strategy for safeguarding endangered alpine flora.",
    "According to a statement issued by the regional hospital authority, emergency surgical wings were upgraded with modern diagnostic imaging units.",
    "Official release from the Ministry of Railways announced completion of track doubling along the hundred-kilometer industrial corridor.",
    "A peer-reviewed article in the New England Journal of Medicine evaluated long-term cardiovascular outcomes in an eight-year cohort study.",
    "Reuters reports that wholesale food grain price indices remained steady across primary agricultural terminal markets.",
    "The European Central Bank confirmed interest rate adjustments aligned with forecasted medium-term inflation target trajectories.",
    "Associated Press reports that national park rangers recorded the first documented sighting of a snow leopard cub in the high sanctuary this spring.",
    "Official gazette notification confirmed the implementation of revised consumer protection rules for cross-border electronic commerce.",
    "BBC News reports that municipal authorities inaugurated a four-hundred-megawatt solar photovoltaic installation on decommissioned industrial land.",
    "According to an announcement by the Indian Space Research Organisation, the solar observation observatory payload completed its calibration phase.",
    "A comprehensive study published in Nature Climate Change quantified the carbon sequestration potential of restored coastal mangrove ecosystems.",
    "Associated Press reported that representatives from thirty nations signed an accord strengthening regulations against illegal maritime fishing.",
    "Reuters reports that the industrial production index increased by 4.2 percent in the preceding quarter driven by capital goods manufacturing.",
    "World Health Organization officials confirmed that the regional polio vaccination campaign achieved ninety-five percent target community coverage.",
]

FAKE_NEWS_AND_SCAM_SAMPLES = [
    # Rural WhatsApp Scams & Welfare Hoaxes
    "URGENT: Government announced Free Solar Scheme 2026! Click here to claim Rs 50000 cash subsidy immediately! Forward before it gets deleted!",
    "ALERT TO ALL VILLAGERS: Free Laptop and Tablet Scheme 2026 approved for every household. Click link http://free-yojana.fake-gov.in to apply now!",
    "ATTENTION: Your electricity power will be cut tonight at 9:30 PM due to pending bill. Call immediately at 9876543210 to prevent disconnection!",
    "WARNING: Gang of 25 child lifters and bacha chor entered our district dressed as sadhus! Keep children inside and forward to all groups immediately!",
    "SHOCKING TRUTH: Doctors hate this simple trick! Drink hot water with lemon and baking soda to cure all cancer in 48 hours guaranteed!",
    "Bank Alert: Your SBI bank account KYC has been suspended today. Click here immediately to update your Aadhaar card or account blocked permanently!",
    "Forward to everyone: NASA confirmed that tonight cosmic rays will pass near earth. Switch off your mobile phones between 12:30 AM to 3:30 AM to avoid radiation!",
    "MIRACLE DISCOVERY: Scientists expose secret that big pharma doesn't want you to know! Boil neem leaves with turmeric to destroy any virus completely!",
    "SHARE WITH EVERYONE: UNESCO just declared the national anthem of our country as the best in the world! Share maximum to make our nation proud!",
    "URGENT WARNING: Poisonous paracetamol tablets containing deadly virus found in local medical stores. Do NOT take any P-500 pills! Share to save lives!",
    "BOMBSHELL: New government order grants Rs 25000 monthly allowance to every unemployed youth. Click this WhatsApp link to register your bank details!",
    "Forwarded as received: If you dial *#06# and send your phone number to this link you will receive 100 GB free internet 5G recharge for one year!",
    "BREAKING NEWS: Mind-blowing secret exposed! Mainstream media is hiding the fact that 5G towers are causing widespread fainting in rural areas!",
    "CRITICAL ALERT: Kidnapping gang active near village schools! They are using black vans to abduct children. Spread this message to every family group!",
    "UNBELIEVABLE: Instant money scheme! Government depositing Rs 5000 directly into all Jan Dhan accounts tonight. Click link to check your status!",
    "Doctors admit: Eating raw onions with salt prevents 100% of heart attacks! Share before they take this video down!",
    "SHOCKING EXPOSE: Deep state conspiracy uncovered! They are secretly putting sterilization chemicals in tap water. Wake up sheeple and share!",
    "Forward to at least 10 people or bad luck for 5 years! Government giving free electric scooters to all students who click here within 10 minutes!",
    "EMERGENCY ADVISORY: Drinking cold water after meals causes instant cancer! Forward to your loved ones immediately to save lives!",
    "Viral audio message: High level officer secretly reveals that entire state will be under complete lockdown starting tomorrow morning at 6 AM!",
    # Satire Outlets (often misread as real news)
    "theonion.com: Department Of Transportation Builds Giant Ramp Over Entire Midwest To Save Drivers Time.",
    "The Babylon Bee reports: Local Church Replaces Pastor With AI Bot Programmed With Thirty-Minute Jokes.",
    "ClickHole: Heartbreaking: The Worst Person You Know Just Made A Great Point.",
    "theonion.com: Economists Warn That If People Keep Saving Money Entire Economy Will Collapse By Tuesday.",
    "The Babylon Bee: Environmentalists Urge Citizens To Stop Exhaling In Order To Curb Carbon Emissions.",
    "theonion.com: Area Man Passionate Defender Of What He Imagines Constitution To Be.",
    "Waterford Whispers News: Local Man Solves Complex Middle Eastern Geopolitics Over Third Pint Of Stout.",
    "theonion.com: New Study Finds Leading Cause Of Brain Cancer Is Staring Directly At Solar Eclipse While Eating Raw Asbestos.",
    "The Babylon Bee: Tech Startup Invents Revolutionary New Way To Walk Somewhere Without Looking At Phone.",
    "theonion.com: Archeologists Unearth 4000-Year-Old To-Do List With Nothing Checked Off.",
    # Additional Disinformation & Clickbait Scams
    "URGENT FORWARD: Free ration card holders will receive free gold coin under new festival scheme! Apply at link before quota ends today!",
    "TERRIFYING TRUTH: Mind-blowing secret exposed! Ancient Himalayan herb cures diabetes in 3 days without any insulin. Big pharma is censoring this!",
    "ALERT: Suspicious people in white ambulance offering free medical injections in village are actually organ trafficking gang! Beat them if seen!",
    "You won't believe what happened! Hidden truth about electricity meters that saves 90 percent bill using a simple magnet. Click here to watch video!",
    "SHARE MAXIMUM: Government launching instant subsidy scheme giving free tractor to every farmer with less than 5 acres. Register your phone number now!",
    "EXPOSED: Mainstream media won't tell you this! Secret cure for all joint pain found in common garden weed. Doctors terrified by this discovery!",
    "Forwarded message: All WhatsApp messages and calls will become chargeable at Rs 5 per day from midnight unless you forward this to 20 contacts!",
    "BOMBSHELL REVELATION: Leaked documents prove climate change is a complete hoax created to sell solar panels. Share before it gets censored!",
    "URGENT NOTICE: Bank account deactivation alert! Your ATM card is expired. Send your 16 digit number and OTP to customer care link immediately!",
    "SHOCKING DISCOVERY: Eating garlic on an empty stomach makes you immune to all biological diseases! Forward to protect every family member!",
    "Child lifter alert: Women begging in disguise are kidnapping young infants from village cradles. Share immediately to every WhatsApp group!",
    "Instant cash reward! You have won Rs 7500000 in international lottery scheme! Send processing fee of Rs 2000 to release fund transfer!",
    "BREAKING EXPOSE: Scientists reveal terrifying truth about microwave ovens that causes instant DNA mutation. Stop using microwaves immediately!",
    "Forward to everyone: Free smartphone yojana 2026 distribution starts today. Click link to verify village voter list and receive handset!",
    "MIRACLE REMEDY: Sniffing mustard oil kills all viruses inside nasal passage instantly according to viral video. Forward to save lives!",
    "URGENT: Government announced complete cancellation of all agricultural bank loans. Click here to download your loan waiver certificate!",
    "Unbelievable miracle! Holy water from local spring cures blindness and paralysis within 24 hours. Forward before authorities shut it down!",
    "SHOCKING BOMBSHELL: Proof that vaccines contain magnetic microchips that track your exact location via satellite. Wake up and refuse!",
    "Free electric bicycle scheme for all rural youth! Apply now with your Aadhaar number at http://pm-free-cycle-yojana.scam.net!",
    "ALERT TO PARENTS: Gang using toxic sweets to drug schoolchildren at gate! Share this alert with every parent and school group right now!",
]


def build_and_save_dataset():
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench_dir = os.path.join(data_dir, "data", "benchmark", "news")
    test_dir = os.path.join(data_dir, "data", "test_benchmark", "news")

    os.makedirs(bench_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Compile dataset with metadata
    all_real = [{"text": s, "label": 0, "category": "authoritative_news"} for s in REAL_NEWS_SAMPLES]
    all_fake = [{"text": s, "label": 1, "category": "disinformation_scam"} for s in FAKE_NEWS_AND_SCAM_SAMPLES]

    # Split: 80% Train, 20% Holdout Test
    train_real = all_real[:40]
    test_real = all_real[40:]

    train_fake = all_fake[:40]
    test_fake = all_fake[40:]

    train_data = train_real + train_fake
    test_data = test_real + test_fake

    train_file = os.path.join(bench_dir, "fake_news_train.json")
    test_file = os.path.join(test_dir, "fake_news_test.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"Successfully generated News & Disinformation Benchmark:")
    print(f"  • Train Set: {len(train_data)} samples -> {train_file}")
    print(f"  • Test Set:  {len(test_data)} samples -> {test_file}")


if __name__ == "__main__":
    build_and_save_dataset()
