import re

drugs = frozenset(
    [
        "clorfenamina",
        "paracetamol", "acetaminofen",
        "ambroxol",
        "diclofenaco",
        "cloranfenicol", "cloramfenicol",
        "ampicilina",
        "trimetoprima",
        "sulfametoxazol",
        "ceftriaxona",
        "ciprofloxacino",
        "benzonatato",
        "biomesina",
        "lutropen",
        "abacavir", "abiraterona", "acarbosa", "aceclofenaco", "acemetacina", "acenocumarol", "acetanilida", "acetilsalicilato", "acetilsalicílico", "acetónido", "acexamato", "acexámico", "aciclovir", "aclidinio", "ademetionina", "adenosina", "albendazol", "alcaftadina", "alendrónico", "alfacalcidol", "alfadihidroergocriptina", "alfuzosina", "aliskireno", "almagato", "alogliptina", "aloina", "alopurinol", "alprazolam", "alprostadil", "aluminio", "alverina", "amantadina", "ambrisentán", "ambroxol", "amfotericina", "amilorida", "aminofenazona", "aminofilina", "amiodarona", "amisulprida", "amitriptilina", "amlodipino", "amlopidino", "amlotriptán", "amoxicilina", "ampicilina", "anastrozol", "anfebutamona", "anfepramona", "anfotericina", "anidulafungina", "aprepitant", "aripiprazol", "ascórbico", "aspartato", "aspergillus", "atenolol", "atomoxetina", "atorvastatina", "atosiban", "atropina", "azacitidina", "azatioprina", "azelaico", "azelastina", "azitromicina", "bazedoxifeno", "beclometasona", "bencidamina", "bencilo", "bencilpenicilina", "benserazida", "benzocaina", "benzonatato", "besilato", "betahistina", "betametasona", "betaxolol", "bicalutamida", "bicarbonato", "bifonazol", "bilastina", "bimatoprost", "biperideno", "bisacodilo", "bismuto", "bisoprolol", "borax", "bórico", "bortezomib", "bosentán", "brimonidina", "brinzolamida", "brivaracetam", "bromazepam", "bromelaína", "bromfeniramina", "bromhexina", "bromhidrato", "bromuro", "budesonida", "bufenina", "bumetanida", "bupivacaina", "buprenorfina", "busulfano", "butenafina", "cabergolina", "cafeina", "cafeína", "cálcica", "cálcico", "calcio", "calcipotriol", "calcitonina", "calcitriol", "candesartan", "caolín", "capecitabina", "captopril", "carbamazepina", "carbazocromo", "carbetocina", "carbidopa", "carbocisteina", "carbón", "carbonato", "carbonilo", "carisoprodol", "carvedilol", "caspofungina", "cefaclor", "cefadroxilo", "cefalexina", "cefdinir", "cefditoren", "cefepima", "cefixima", "cefotaxima", "cefpodoxima", "ceftazidima", "ceftibuteno", "ceftolozano", "ceftriaxona", "cefuroxima", "celecoxib", "celulasa", "cetirizina", "cetrimonio", "cetrorelix", "cianocobalamina", "ciclesonida", "ciclobenzaprina", "ciclofosfamida", "ciclopentolato", "ciclopirox", "ciclosporina", "cilastatina", "cilexetilo", "cilostazol", "cimetidina", "cinacalcet", "cinarizina", "cincocaína", "cinitaprida", "ciprofibrato", "ciprofloxacino", "ciproheptadina", "ciproterona", "cisaprida", "cisatracurio", "citalopram", "citarabina", "citicolina", "citidin", "citrato", "citrico", "cítrico", "claritromicina", "clavulanico", "clavulánico", "cleboprida", "clemizol", "clenbuterol", "clindamicina", "clioquinol", "clobazam", "clobenzorex", "clobetasol", "clomifeno", "clomipramina", "clonazepam", "clonidina", "clonixinato", "cloperastina", "clopidogrel", "clorambucilo", "cloranfenicol", "clorfenamina", "clormadinona", "cloropiramina", "cloroquina", "clorpropamida", "clortalidona", "cloruro", "clorzoxazona", "clostridiopeptidasa", "clotrimazol", "clozapina", "cobamamida", "codeína", "colchicina", "colecalciferol", "colestiramina", "colina", "colistimetato", "complejo", "condroitina", "condrotina", "conjugados", "cristalina", "crospovidona", "crotamitón", "cumarina", "dabigatrán", "dalteparina", "danazol", "dapagliflozina", "dapoxetina", "daptomicina", "darifenacina", "darunavir", "decanoato", "deferasirox", "deflazacort", "degarelix", "dehidrocolato", "desflurano", "desloratadina", "desmopresina", "desogestrel", "desvenlafaxina", "dexametasona", "dexketoprofeno", "dexlansoprazol", "dexmedetomidina", "dexphantenol", "dexrazoxano", "dextrometorfano", "diacereina", "diacereína", "diazepam", "dicicloverina", "diclofenaco", "dicloxacilina", "didanosina", "didrogesterona", "dienogest", "difenhidramina", "difenidol", "difenilpiralina", "diflucortolona", "digoxina", "dihexazin", "dihidroergocristina", "diiodohidroxiquinoleina", "diltiazem", "dimenhidrinato", "dimeticona", "dimetilfumarato", "dinoprostona", "diosmina", "dipiridamol", "disódico", "disopiramida", "disoproxilo", "disulfiram", "diyodohidroxiquinoleina", "diyodohidroxiquinoleína", "dobesilato", "docetaxel", "docusato", "dolasetrón", "domestica", "domperidona", "donepezilo", "dorzolamida", "doxazosina", "doxepina", "doxiciclina", "doxilamina", "doxofilina", "doxorubicina", "dronedarona", "dropropizina", "drospirenona", "drotaverina", "duloxetina", "dutasterida", "ebastina", "efavirenz", "elemental", "eletriptán", "emtricitabina", "enalapril", "enfuvirtida", "enoxaparina", "enoxolona", "entacapona", "entecavir", "epinastina", "epinefrina", "epirubicina", "eplerenona", "eprosartán", "equivalente", "erdosteína", "ergotamina", "eribulina", "eritromicina", "erlotinib", "ertapenem", "escitalopram", "esomeprazol", "espiramicina", "espironolactona", "estazolam", "ésteres", "estradiol", "estramustina", "estreptodornasa", "estreptomicina", "estreptoquinasa", "estriol", "estrogenos", "estrógenos", "estroncio", "etambutol", "etamsilato", "etanbutol", "etilefrina", "etílicos", "etilo", "etinilestradiol", "etofenamato", "etonogestrel", "etoricoxib", "etosuximida", "etravirina", "everolimus", "exemestano", "exenatida", "extracto", "ezetimiba", "famotidina", "felodipino", "fenacetina", "fenazopiridina", "fenilbutazona", "fenilefrina", "fenilpropionato", "feniramina", "fenitoina", "fenitoína", "fenobarbital", "fenofibrato", "fenolftaleína", "fenoterol", "fenotrina", "fenoverina", "fenproporex", "fentanilo", "fentermina", "fenticonazol", "ferrico", "ferroso", "fexofenadina", "finasterida", "fingolimod", "flavoxato", "flecainida", "floroglucinol", "fluconazol", "fludarabina", "flufenazina", "flunarizina", "fluocinolona", "fluocinonida", "fluorometolona", "fluorouracilo", "fluoxetina", "flupentixol", "flurbiprofeno", "fluticasona", "flutrimazol", "fluvoxamina", "fólico", "fondaparinux", "formoterol", "fosamprenavir", "fosaprepitant", "fosfato", "fosfolípidos", "fosfomicina", "ftalilsulfatiazol", "fulvestrant", "fumarato", "furosemida", "fusidato", "fusídico", "gabapentina", "galantamina", "gatifloxacino", "gefitinib", "gelatina", "gemcitabina", "gemfibrozilo", "gemifloxacino", "gentamicina", "gestodeno", "gestonorona", "glatiramer", "glibenclamida", "glicerilo", "glicerol", "glicina", "gliclazida", "glicofosfopeptical", "glicopirronio", "glimepirida", "glipizida", "glucosa", "glucosamina", "goserelina", "granisetrón", "grasos", "guaifenesina", "guayacol", "haloperidol", "hemezol", "hemihidratado", "hesperidina", "hidralazina", "hidroclorotiazida", "hidroclotiazida", "hidrocortisona", "hidroquinona", "hidrosmina", "hidrotalcita", "hidroxicarbamida", "hidroxicloroquina", "hidroxicobalamina", "hidróxido", "hidroxiprogesterona", "hidroxipropilteofilina", "hidroxitetracloruro", "hidroxizina", "hidroxocobalamina", "hierro", "hioscina", "hipromelosa", "ibandrónico", "ibesartán", "ibuprofeno", "idarubicina", "idebenona", "iloprost", "imatinib", "imipenem", "imipramina", "imiquimod", "indacaterol", "indapamida", "indobufeno", "indometacina", "inosina", "iodo", "ipratropio", "irbesartan", "irinotecan", "isocaproato", "isoconazol", "isoniazida", "isopropamidayoduro", "isosorbida", "isotipendilo", "itoprida", "itraconazol", "ivabradina", "ivermectina", "ketanserina", "ketoconazol", "ketoprofeno", "ketorolaco", "ketotifeno", "lacidipino", "lacosamida", "lactato", "lactulosa", "lamivudina", "lamotrigina", "lanreotida", "lansoprazol", "lapatinib", "laropiprant", "latanoprost", "laurilsulfato", "leflunomida", "lenalidomida", "lercanidipino", "letrozol", "leuprorelina", "levetiracetam", "levocarnitina", "levocetirizina", "levodopa", "levodropropizina", "levofloxacino", "levomepromazina", "levonorgestrel", "levopantoprazol", "levosimendan", "levosulpirida", "levotiroxina", "lidamidina", "lidocaina", "lidocaína", "limeciclina", "linagliptina", "lincomicina", "linezolid", "liotironina", "liposomal", "lisina", "lisinopril", "litio", "loflazepato", "loperamida", "lopinavir", "loratadina", "lorazepam", "losartan", "losartán", "loteprednol", "loxoprofeno", "macitentán", "macrogol", "magaldrato", "magnesio", "manganeso", "manitol", "maraviroc", "mazindol", "mebendazol", "mebeverina", "meclozina", "medoxomilo", "medroxiprogesterona", "mefenámico", "melatonina", "melfalán", "meloxicam", "memantina", "menadiona", "mentol", "mercaptopurina", "meropenem", "mesalazina", "mesilato", "mesna", "mesterolona", "mestranol", "metadona", "metadoxina", "metamizol", "metenamina", "metformina", "metildopa", "metilfenidato", "metilo", "metilprednisolona", "metisoprinol", "metocarbamol", "metoclopramida", "metoprolol", "metotrexato", "metoxaleno", "metoxicinamato", "metronidazol", "mianserina", "micofenólico", "miconazol", "midazolam", "mifepristona", "milrinona", "minociclina", "mirtazapina", "misoprostol", "moclobemida", "modafinilo", "moexipril", "mometasona", "monohidratado", "montelukast", "morfina", "mosaprida", "moxifloxacino", "mupirocina", "nabilona", "nadifloxacino", "nadroparina", "naltrexona", "nandrolona", "naproxeno", "nebivolol", "neomicina", "neostigmina", "nepafenaco", "nevirapina", "nicergolina", "nicotina", "nicotinico", "nicotínico", "nifedipino", "nifuratel", "nifuroxazida", "nilotinib", "nimodipino", "nistatina", "nitazoxanida", "nitrofurantoina", "nomegestrol", "norelgestromina", "noretisterona", "norfenilefrina", "norfloxacino", "norgestrel", "norpseudoefedrina", "nortriptilina", "octreotida", "ofloxacino", "olamina", "olanzapina", "olmesartán", "olopatadina", "omega", "omeprazol", "ondansetrón", "orciprenalina", "orfenadrina", "ornitina", "oseltamivir", "otilonio", "oxaliplatino", "oxcarbazepina", "oxeladina", "oxibenzona", "oxibutinina", "oxicodona", "oxido", "oximetazolina", "oxitetraciclina", "padimato", "palbociclib", "palonosetrón", "pamabrom", "pancreatina", "pantoprazol", "pantotenato", "papaína", "papaverina", "paracetamol", "parametasona", "parecoxib", "pargeverina", "paricalcitol", "paroxetina", "pazopanib", "pectina", "pemetrexed", "pentoxifilina", "perfenazina", "perindopril", "permetrina", "picosulfato", "pidotimod", "piketoprofeno", "pimecrolimus", "pinaverio", "pinazepam", "pindolol", "pioglitazona", "pipazetato", "pipemídico", "piperacilina", "piperidolato", "piracetam", "pirantel", "pirazinamida", "piridostigmina", "piridoxina", "pirilamina", "pirimetamina", "piritinol", "piroxicam", "pitavastatina", "pivoxilo", "plata", "policresuleno", "polidocanol", "poliestireno", "polimaltosado", "polimixina", "porcino", "posaconazol", "potásico", "potasio", "pramipexol", "pranobex", "prasterona", "prazicuantel", "prazosina", "prednicarbato", "prednisolona", "prednisona", "pregabalina", "pridinol", "prilocaina", "primidona", "probenecida", "procaina", "procainica", "progesterona", "promestrieno", "propafenona", "propanodiol", "propanolol", "propionato", "propofol", "proxifilina", "prucaloprida", "prunus", "pulmon", "quetiapina", "quinagolida", "quinapril", "quinfamida", "rabeprazol", "racecadrotilo", "raloxifeno", "raltegravir", "ramipril", "ranelato", "rasagilina", "remifentanilo", "retinol", "ribavirina", "rifamicina", "rifampicina", "rifaximina", "riluzole", "rimantadina", "risedronato", "risperidona", "ritonavir", "rivaroxabán", "rivastigmina", "rizatriptan", "rocuronio", "rosiglitazona", "rosuvastatina", "rotigotina", "rupatadina", "sacarato", "salbutamol", "salicilato", "salicílico", "salicitato", "salmeterol", "s-amlodipino", "saxagliptina", "secnidazol", "selegilina", "senósidos", "serratiopeptidasa", "sertaconazol", "sertralina", "sevelámero", "sevoflurano", "sildenafil", "silimarina", "simeticona", "simvastatina", "sirolimus", "sitagliptina", "sodio", "sódio", "solifenacina", "sorafenib", "sucralfato", "sugammadex", "sulbactam", "sulbutiamina", "sulfacetamida", "sulfadiazina", "sulfametoxazol", "sulfasalazina", "sulfoguayacol", "sulfonato", "sulindaco", "sulodexida", "sulpirida", "sumatriptan", "sunitinib", "tacrolimus", "tadalafil", "tafluprost", "talidomida", "tamoxifeno", "tamsulosina", "tanato", "tapentadol", "tartárico", "tazobactam", "tedizolid", "tegaserod", "teicoplanina", "telmisartan", "telmisartán", "temozolamida", "temsirolimus", "tenofovir", "teofilina", "terbinafina", "terbutilamina", "terconazol", "teriflunomida", "testosterona", "tetracaina", "tetraciclina", "tetrizolina", "tiamazol", "tiamina", "tianfenicol", "tiaprofénico", "tibezonio", "tibolona", "ticagrelor", "tigeciclina", "timolol", "tinidazol", "tiocolchicósido", "tioconazol", "tióctico", "tiotropio", "tizanidina", "tobramicina", "tolperisona", "tolterodina", "topiramato", "trabectedina", "tramadol", "trandolapril", "travoprost", "trazodona", "tretinoina", "tretinoína", "triamcinolona", "triazolam", "tribenósido", "triclosan", "trietanolamina", "trifluoperazina", "triflusal", "trihexifenidilo", "trimebutina", "trimetazidina", "trimetilfloroglucinol", "trimetoprima", "trinitrato", "triptorelina", "triyodotironina", "trometamina", "trometamol", "troxerutina", "undecilenato", "undecilénico", "undecilinato", "uridin", "ursodeoxicólico", "valaciclovir", "valganciclovir", "valproato", "valproico", "valsartan", "valsartán", "vardenafil", "vareniclinatartrato", "venlafaxina", "verapamilo", "vigabatrina", "vildagliptina", "vinorelbina", "vitamina", "voriconazol", "vorinostat", "warfarina", "yodocaseina", "yoduro", "zafirlukast", "zanamivir", "zidovudina", "zinc", "ziprasidona", "zirconio", "zofenopril", "zoledronico", "zolmitriptano", "zolpidem", "zuclopentixol"
    ]
)

amounts = frozenset(
    [
        "1", "un", "una",
        "2", "dos",
        "3", "tres",
        "4", "cuatro",
        "5", "cinco",
        "6", "seis",
        "7", "siete",
        "8", "ocho",
        "9", "nueve",
        "10", "diez",
        "11", "once",
        "12", "doce",
        "13", "trece",
        "14", "catorce",
        "15", "quince",
        "16", "dieciseis", "dieciséis",
        "17", "diecisiete",
        "18", "dieciocho",
        "19", "diecinueve",
        "20", "veinte",
        "21", "veintiuno",
        "22", "veintidos", "veintidós",
        "23", "veintitres",
        "24", "veinticuatro",
        "25", "veinticinco",
        "26", "veintiseis",
        "27", "veintisiete",
        "28", "veintiocho",
        "29", "veintinueve",
        "30", "treinta",
    ]
)

unities = frozenset(
    [
        "capsula", "capsulas",
        "mililitro", "mililitros", "ml",
        "tableta", "tabletas",
        "g", "gramos",
        "cucharada", "cucharadas",
        "gotas", "gota",
        "ampolleta", "ampolletas",
        "sobre", "sobres",
        "inhalacion", "inhalaciones",
        "ámpula", "ampula",
        "dosis",
        "pastilla", "pastillas",
        "aplicación", "aplicacion", "aplicaciones"
    ]
)

frequencies = frozenset(
    [
        "hora", "horas",
    ]
)

periods = frozenset(
    [
        "dia", "dias", "día", "días",
    ]
)

def clean_text(corpus):

    new_corpus = []
    temp = ""

    corpus = corpus.lower()
    corpus = corpus.strip()

    to_replace = ["!", "?", ".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "/", "\\", "|", "°", "º", "ª", "`", "¨", "~", "-", "_", "'", '"']
    for char in to_replace: corpus = corpus.replace(char, "")

    for text in corpus.split("\n"):
        if text.strip() != "":
            temp += text + " "
        else:
            new_corpus.append(temp)
            temp = ""

    if temp != "":
        new_corpus.append(temp)

    text = " ".join(new_corpus)
    text = re.sub(r"\s+", " ", text)
    
    sentences = re.split(r"(\d{4,})", text)
    
    # Join sentences that have one word with the next sentence
    for i in range(len(sentences)):
        if i != 0 and len(sentences[i].split(" ")) == 1:
            sentences[i - 1] += sentences[i]
            sentences[i] = ""

    sentences = [sentence for sentence in sentences if sentence != ""]

    for sentence in sentences:
        if len(sentence.split(" ")) < 10:
            sentences.remove(sentence)
    
    return sentences

def get_text_indications(text):

    sentences = clean_text(text)
    indications = []

    for sentence in sentences:

        state = 1

        drug = None
        amount1 = None
        unity = None
        amount2 = None
        frequency = None
        amount3 = None
        period = None

        for token in sentence:

            if token in drugs:

                drug = token
                state = 2

            elif token in amounts:
                
                amount1 = amount2
                amount2 = amount3
                amount3 = token

                if state == 1 or state > 6:
                    state = 1
                elif state == 4:
                    state = 5
                elif state == 6:
                    state = 7
                else:
                    state = 3

            elif token in unities:
                
                unity = token

                if state == 2 or state == 4:
                    state = 2
                elif state == 3 or state == 5:
                    state = 4
                else: 
                    state = 1

            elif token in frequencies:

                frequency = token

                if state == 5:
                    state = 6
                else:
                    state = 1

            elif token in periods:
                
                period = token

                if state == 7:
                    state = 8
                else:
                    state = 1

            else:

                if state == 5:
                    state = 3

                elif state == 7 or state == 8:
                    state = 1

        indications.append({
            "drug": drug,
            "amount1": amount1,
            "unity": unity,
            "amount2": amount2,
            "frequency": frequency,
            "amount3": amount3,
            "period": period
        })

    print(indications)

    return indications