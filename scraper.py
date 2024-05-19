import json
import re
import requests
from bs4 import BeautifulSoup

def clean_text(text):
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Удаление символов-разделителей
    text = re.sub(r'[^а-яА-Я0-9\s]', '', text)
    
    return text

def clean_recipe_data(recipe_data):
    # Очистка названия рецепта
    recipe_data['title'] = clean_text(recipe_data['title'])
    
    # Очистка ингредиентов
    recipe_data['ingredients'] = [clean_text(ingredient) for ingredient in recipe_data['ingredients']]
    
    # Очистка инструкций
    recipe_data['instructions'] = [clean_text(instruction) for instruction in recipe_data['instructions']]
    
    return recipe_data


def get_recipe_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    title_element = soup.title
    if title_element:
        title = title_element.string.strip()
    else:
        title = ""

    ingredients_elements = soup.select('ul.ingredients__list > li')
    ingredients = [ingredient.text.strip() for ingredient in ingredients_elements]

    instructions_elements = soup.select('div.recipe-rich > ol > li > span[itemprop="text"]')
    instructions = [instruction.text.strip() for instruction in instructions_elements]

    # Извлечение категории рецепта
    category_elements = soup.select('ul.masthead-catalog__list > li > a')
    categories = [category_element.text.strip().lower() for category_element in category_elements]
    categories = [category.replace('все рецепты', '') for category in categories]

    category = next((category for category in categories if category in url.lower()), None)

    if category is None:
        category = "Другое"

    return {
        'title': title,
        'ingredients': ingredients,
        'instructions': instructions,
        'category': category
    }

# Очистка данных рецепта
    recipe_data = clean_recipe_data(recipe_data)

    return recipe_data

def main():
    recipe_links = [
        'https://www.vsegdavkusno.ru/recipes/myasnaya-sbornaya-solyanka',
        'https://www.vsegdavkusno.ru/recipes/tushenaya-kapusta',
        'https://www.vsegdavkusno.ru/recipes/marinovannyy-luk-za-25-minut',
        'https://www.vsegdavkusno.ru/recipes/chechevichnyy-sup',
        'https://www.vsegdavkusno.ru/recipes/lapsha-udon-s-kuricey-v-souse-teriyaki',
        'https://www.vsegdavkusno.ru/recipes/sup-s-frikadelkami',
        'https://www.vsegdavkusno.ru/recipes/hachapuri-po-adzharski',
        'https://www.vsegdavkusno.ru/recipes/domashniy-mayonez-za-1-minutu',
        'https://www.vsegdavkusno.ru/recipes/chebureki',
        'https://www.vsegdavkusno.ru/recipes/pshenichnaya-kasha',
        'https://www.vsegdavkusno.ru/recipes/oladi-iz-kabachkov-s-sousom-dzadziki-grecheskaya-kuhnya',
        'https://www.vsegdavkusno.ru/recipes/sendvich-s-yaycom-i-vetchinoy',
        'https://www.vsegdavkusno.ru/recipes/podliva-iz-farsha',
        'https://www.vsegdavkusno.ru/recipes/kukuruznaya-kasha',
        'https://www.vsegdavkusno.ru/recipes/xashbraun',
        'https://www.vsegdavkusno.ru/recipes/lepeshka-s-nachinkoj-na-skovorode',
        'https://www.vsegdavkusno.ru/recipes/vozdushnyij-omlet-s-tvorozhnyim-syirom-i-pomidorami',
        'https://www.vsegdavkusno.ru/recipes/pshennaya-kasha',
        'https://www.vsegdavkusno.ru/recipes/goryachij-buterbrod-na-skovorode',
        'https://www.vsegdavkusno.ru/recipes/baursaki',
        'https://www.vsegdavkusno.ru/recipes/sendvich-monte-kristo',
        'https://www.vsegdavkusno.ru/recipes/tonkie-bliny-iz-zavarnogo-testa-na-moloke',
        'https://www.vsegdavkusno.ru/recipes/tvorozhno-ovoshchnye-keksy',
        'https://www.vsegdavkusno.ru/recipes/gollandskiy-blin',
        'https://www.vsegdavkusno.ru/recipes/bulgur-s-ovoshchami',
        'https://www.vsegdavkusno.ru/recipes/skrembl-yaichnica-boltunya',
        'https://www.vsegdavkusno.ru/recipes/brizol',
        'https://www.vsegdavkusno.ru/recipes/risovaya-kasha-na-moloke',
        'https://www.vsegdavkusno.ru/recipes/tykvennyy-humus',
        'https://www.vsegdavkusno.ru/recipes/denverskij-omlet-s-vetchinoj-i-syirom-v-duxovke',
        'https://www.vsegdavkusno.ru/recipes/kruassanyi-iz-gotovogo-sloenogo-testa',
        'https://www.vsegdavkusno.ru/recipes/lepeshki-kong-yu-bing',
        'https://www.vsegdavkusno.ru/recipes/ovsyanaya-kasha',
        'https://www.vsegdavkusno.ru/recipes/bananovyie-oladi',
        'https://www.vsegdavkusno.ru/recipes/chizkejk-bez-vyipechki',
        'https://www.vsegdavkusno.ru/recipes/mish-mash',
        'https://www.vsegdavkusno.ru/recipes/vkusnyie-lenivyie-vareniki',
        'https://www.vsegdavkusno.ru/recipes/volshebnyij-franczuzskij-tost',
        'https://www.vsegdavkusno.ru/recipes/sloppy-joe',
        'https://www.vsegdavkusno.ru/recipes/menemen',
        'https://www.vsegdavkusno.ru/recipes/omlet-s-lisichkami',
        'https://www.vsegdavkusno.ru/recipes/yajcza-orsini',
        'https://www.vsegdavkusno.ru/recipes/yablochnyij-krambl',
        'https://www.vsegdavkusno.ru/recipes/40-sekund-omlet',
        'https://www.vsegdavkusno.ru/recipes/vozdushnyij-omlet-sufle',
        'https://www.vsegdavkusno.ru/recipes/czvetaevskij-yablochnyij-pirog',
        'https://www.vsegdavkusno.ru/recipes/buzhenina',
        'https://www.vsegdavkusno.ru/recipes/yablochnyij-chudo-pirog-za-10-minut',
        'https://www.vsegdavkusno.ru/recipes/nezhnyie-syirniki-na-skovorode',
        'https://www.vsegdavkusno.ru/recipes/chirbuli-po-adzharski',
        'https://www.vsegdavkusno.ru/recipes/bananovyij-tort',
        'https://www.vsegdavkusno.ru/recipes/burrito-s-yajczom-i-bekonom',
        'https://www.vsegdavkusno.ru/recipes/panini-s-kuriczej-i-syirom',
        'https://www.vsegdavkusno.ru/recipes/syitnyij-omlet',
        'https://www.vsegdavkusno.ru/recipes/semejnyij-goryachij-buterbrod',
        'https://www.vsegdavkusno.ru/recipes/yaichnicza-s-pomidorami-po-marokkanski',
        'https://www.vsegdavkusno.ru/recipes/domashnyaya-kurinaya-kolbasa',
        'https://www.vsegdavkusno.ru/recipes/blinyi-po-czarski-s-myasom-i-sousom-beshamel',
        'https://www.vsegdavkusno.ru/recipes/moldavskij-skrob-s-bryinzoj',
        'https://www.vsegdavkusno.ru/recipes/morkovnyie-kotletyi',
        'https://www.vsegdavkusno.ru/recipes/tradiczionnaya-tvorozhnaya-zapekanka',
        'https://www.vsegdavkusno.ru/recipes/tostyi-s-yajczom',
        'https://www.vsegdavkusno.ru/recipes/omlet-s-ovoshhami-po-italyanski',
        'https://www.vsegdavkusno.ru/recipes/yablochnye-oladi',
        'https://www.vsegdavkusno.ru/recipes/yajcza-pashot',
        'https://www.vsegdavkusno.ru/recipes/mannaya-kasha',
        'https://www.vsegdavkusno.ru/recipes/yablochnyij-tort-na-skovorode',
        'https://www.vsegdavkusno.ru/recipes/sendvichi-s-kuriczej-dilli',
        'https://www.vsegdavkusno.ru/recipes/omlet-s-ovoshhami',
        'https://www.vsegdavkusno.ru/recipes/myasnyie-pirozhki-iz-sloenogo-testa',
        'https://www.vsegdavkusno.ru/recipes/pashtet-iz-seledki',
        'https://www.vsegdavkusno.ru/recipes/yablochnyij-pirog-iz-sloenogo-testa',
        'https://www.vsegdavkusno.ru/recipes/xarissa',
        'https://www.vsegdavkusno.ru/recipes/palak-panir',
        'https://www.vsegdavkusno.ru/recipes/tushenaya-kapusta-po-indiyski',
        'https://www.vsegdavkusno.ru/recipes/bitochki-iz-cvetnoy-kapusty',
        'https://www.vsegdavkusno.ru/recipes/utka-zapechennaya-s-yablokami',
        'https://www.vsegdavkusno.ru/recipes/draniki',
        'https://www.vsegdavkusno.ru/recipes/kartofel-po-bombeyski',
        'https://www.vsegdavkusno.ru/recipes/graten-iz-kartofelya',
        'https://www.vsegdavkusno.ru/recipes/shaurma-s-kuricey',
        'https://www.vsegdavkusno.ru/recipes/svinye-rebryshki',
        'https://www.vsegdavkusno.ru/recipes/oladi-iz-kabachkov',
        'https://www.vsegdavkusno.ru/recipes/ovoshchi-na-grile',
        'https://www.vsegdavkusno.ru/recipes/sladkiy-perec-farshirovannyy-syrom',
        'https://www.vsegdavkusno.ru/recipes/zapechnyy-kartofel-farshirovannyy-kurinoy-grudkoy',
        'https://www.vsegdavkusno.ru/recipes/hasselbek-potatoes',
        'https://www.vsegdavkusno.ru/recipes/krevetki-saganaki',
        'https://www.vsegdavkusno.ru/recipes/sacivi-s-kuricey',
        'https://www.vsegdavkusno.ru/recipes/syrnye-lepeshki-na-skovorode',
        'https://www.vsegdavkusno.ru/recipes/posikunchiki-s-myasom',
        'https://www.vsegdavkusno.ru/recipes/slivochnyy-kartofelnyy-graten',
        'https://www.vsegdavkusno.ru/recipes/zharenaya-kartoshka',
        'https://www.vsegdavkusno.ru/recipes/zapekanka-iz-kabachkov',
        'https://www.vsegdavkusno.ru/recipes/belyashi',
        'https://www.vsegdavkusno.ru/recipes/rublennye-kotlety-iz-kurinoy-grudki',
        'https://www.vsegdavkusno.ru/recipes/kartofelnye-chipsy-zavitki-s-syrom',
        'https://www.vsegdavkusno.ru/recipes/aromatnaya-svinaya-koreyka-zapechennaya-s-travami',
        'https://www.vsegdavkusno.ru/recipes/kurinye-zheludki-po-koreyski',
        'https://www.vsegdavkusno.ru/recipes/oladi-na-kefire-s-karamelno-slivochnym-sousom',
        'https://www.vsegdavkusno.ru/recipes/bliny-na-kefire',
        'https://www.vsegdavkusno.ru/recipes/pita-po-meksikanski',
        'https://www.vsegdavkusno.ru/recipes/kurinye-zheludki-po-kitayski',
        'https://www.vsegdavkusno.ru/recipes/fajitas-fahitas-s-kuricey',
        'https://www.vsegdavkusno.ru/recipes/kakuni-tushenaya-svinaya-grudinka',
        'https://www.vsegdavkusno.ru/recipes/pirozhki-s-myasom-na-skovorode',
        'https://www.vsegdavkusno.ru/recipes/kartofelnye-dolki',
        'https://www.vsegdavkusno.ru/recipes/kartofelnyy-kugel',
        'https://www.vsegdavkusno.ru/recipes/tonkie-bliny-iz-zavarnogo-testa-na-moloke',
        'https://www.vsegdavkusno.ru/recipes/pyure-iz-kartofelya-i-luka-poreya',
        'https://www.vsegdavkusno.ru/recipes/tvorozhno-ovoshchnye-keksy',
        'https://www.vsegdavkusno.ru/recipes/shotlandskiy-kartofelnyy-pirog-zapekanka',
        'https://www.vsegdavkusno.ru/recipes/kuskus-so-svezhimi-ovoshhami',
        'https://www.vsegdavkusno.ru/recipes/pasta-s-krevetkami-v-slivochnom-souse',
        'https://www.vsegdavkusno.ru/recipes/kroshka-kartoshka',
        'https://www.vsegdavkusno.ru/recipes/ryiba-pod-marinadom',
        'https://www.vsegdavkusno.ru/recipes/kuricza-v-apelsinovom-souse',
        'https://www.vsegdavkusno.ru/recipes/kurinyie-kryilyishki-dvojnoj-obzharki',
        'https://www.vsegdavkusno.ru/recipes/zharenyie-ovoshhi',
        'https://www.vsegdavkusno.ru/recipes/zharenyie-shampinonyi',
        'https://www.vsegdavkusno.ru/recipes/sosiski-v-klyare',
        'https://www.vsegdavkusno.ru/recipes/svinyie-rebryishki-v-souse-adobo',
        'https://www.vsegdavkusno.ru/recipes/yajcza-pashot',
        'https://www.vsegdavkusno.ru/recipes/svinyie-rebryishki-v-duxovke',
        'https://www.vsegdavkusno.ru/recipes/xashbraun',
        'https://www.vsegdavkusno.ru/recipes/baklazhanyi-po-yaponski',
        'https://www.vsegdavkusno.ru/recipes/nemeckiy-kartofelnyy-salat',
        'https://www.vsegdavkusno.ru/recipes/kartoshka-po-derevenski',
        'https://www.vsegdavkusno.ru/recipes/imam-bayaldyi',
        'https://www.vsegdavkusno.ru/recipes/frikase-iz-kuriczyi',
        'https://www.vsegdavkusno.ru/recipes/kurinoe-file-s-bolgarskim-perczem',
        'https://www.vsegdavkusno.ru/recipes/soba-s-kuriczej',
        'https://www.vsegdavkusno.ru/recipes/baklazhanyi-po-grecheski',
        'https://www.vsegdavkusno.ru/recipes/gushtnut',
        'https://www.vsegdavkusno.ru/recipes/xorovacz',
        'https://www.vsegdavkusno.ru/recipes/ragu-iz-ovoshhej-s-govyadinoj-po-gruzinski',
        'https://www.vsegdavkusno.ru/recipes/banosh',
        'https://www.vsegdavkusno.ru/recipes/bograch',
        'https://www.vsegdavkusno.ru/recipes/zapekanka-iz-kabachkov-v-syirnom-souse',
        'https://www.vsegdavkusno.ru/recipes/zapechennaya-ryiba-s-kartofelem-po-grecheski',
        'https://www.vsegdavkusno.ru/recipes/lapsha-udon-s-govyadinoj-ovoshhami-i-gribami',
        'https://www.vsegdavkusno.ru/recipes/kazan-kebab',
        'https://www.vsegdavkusno.ru/recipes/kurinyie-okorochka-s-ovoshhami',
        'https://www.vsegdavkusno.ru/recipes/ostri',
        'https://www.vsegdavkusno.ru/recipes/legkoe-zharkoe-iz-kartofelya',
        'https://www.vsegdavkusno.ru/recipes/nan-palau',
        'https://www.vsegdavkusno.ru/recipes/farshirovannyie-baklazhanyi-po-tureczki',
        'https://www.vsegdavkusno.ru/recipes/kotletyi-iz-govyadinyi',
        'https://www.vsegdavkusno.ru/recipes/svinyie-otbivnyie-s-gribami',
        'https://www.vsegdavkusno.ru/recipes/paprikash-iz-kurinyix-serdechek-i-pecheni',
        'https://www.vsegdavkusno.ru/recipes/ossobuko',
        'https://www.vsegdavkusno.ru/recipes/nastoyashhee-azu-po-tatarski',
        'https://www.vsegdavkusno.ru/recipes/svinina-s-kvashenoj-kapustoj-v-duxovke',
        'https://www.vsegdavkusno.ru/recipes/chicken-adobo',
        'https://www.vsegdavkusno.ru/recipes/dublin-coddle-irlandskoe-ragu-iz-kartofelya-i-kolbasyi',
        'https://www.vsegdavkusno.ru/recipes/baranina-s-kuskusom-po-marokkanski',
        'https://www.vsegdavkusno.ru/recipes/svinina-adobo',
        'https://www.vsegdavkusno.ru/recipes/kurinyie-bedra-s-ovoshhami-v-duxovke',
        'https://www.vsegdavkusno.ru/recipes/kurinaya-grudka-v-souse-pesto',
        'https://www.vsegdavkusno.ru/recipes/kotletyi-iz-sudaka',
        'https://www.vsegdavkusno.ru/recipes/czvetnaya-kapusta-v-souse-tofu',
        'https://www.vsegdavkusno.ru/recipes/stejk-machete-s-sousom-chimichurri',
        'https://www.vsegdavkusno.ru/recipes/svinina-s-ovoshhami-v-slivochnom-souse',
        'https://www.vsegdavkusno.ru/recipes/kuricza-po-serbski',
        'https://www.vsegdavkusno.ru/recipes/zapekanka-iz-kartofelya-i-kapustyi',
        'https://www.vsegdavkusno.ru/recipes/kurinaya-pechen-na-skovorode-po-vengerski',
        'https://www.vsegdavkusno.ru/recipes/makaronyi-alla-vodka',
        'https://www.vsegdavkusno.ru/recipes/lapsha-yakisoba',
        'https://www.vsegdavkusno.ru/recipes/zharenaya-kapusta-s-sosiskami-i-bekonom',
        'https://www.vsegdavkusno.ru/recipes/kuricza-s-zelenoj-fasolyu-po-armyanski',
        'https://www.vsegdavkusno.ru/recipes/pozharskie-kotletyi',
        'https://www.vsegdavkusno.ru/recipes/bulgur-palau',
        'https://www.vsegdavkusno.ru/recipes/ovoshhi-na-skovorode-v-aziatskom-stile',
        'https://www.vsegdavkusno.ru/recipes/tushenyij-kartofel-so-svininoj',
        'https://www.vsegdavkusno.ru/recipes/zharenyij-ris-s-bekonom-po-aziatski',
        'https://www.vsegdavkusno.ru/recipes/ovoshhnoe-ragu',
        'https://www.vsegdavkusno.ru/recipes/farshirovannyie-kabachki',
        'https://www.vsegdavkusno.ru/recipes/svinyie-rebryishki-v-souse-adobo',
        'https://www.vsegdavkusno.ru/recipes/kuricza-margarita',
        'https://www.vsegdavkusno.ru/recipes/svinyie-otbivnyie-s-grushej-i-batatom',
        'https://www.vsegdavkusno.ru/recipes/ratatuj-na-skovorode',
        'https://www.vsegdavkusno.ru/recipes/chaxoxbili-v-kazane',
        'https://www.vsegdavkusno.ru/recipes/kurinyie-bedra-na-skovorode',
        'https://www.vsegdavkusno.ru/recipes/pasta-s-kuriczej-v-slivochnom-souse',
        'https://www.vsegdavkusno.ru/recipes/kapustnaya-zapekanka-s-farshem',
        'https://www.vsegdavkusno.ru/recipes/rizotto-s-tomatami-i-bazilikom',
        'https://www.vsegdavkusno.ru/recipes/tushenaya-kapusta-s-farshem-na-skovorode',
        'https://www.vsegdavkusno.ru/recipes/golubcy-po-turecki',
        'https://www.vsegdavkusno.ru/recipes/befstroganov-iz-svininy',
        'https://www.vsegdavkusno.ru/recipes/kartofelnaya-zapekanka-s-kuricey-i-gribami',
        'https://www.vsegdavkusno.ru/recipes/kartofelnaya-zapekanka-s-farshem-v-duhovke',
        'https://www.vsegdavkusno.ru/recipes/ris-s-ovoschami',
        'https://www.vsegdavkusno.ru/recipes/kartofelnyy-kugel',
        'https://www.vsegdavkusno.ru/recipes/kartofel-s-gribami-v-slivochnom-souse',
        'https://www.vsegdavkusno.ru/recipes/file-kurinoy-grudke-v-smetane',
        'https://www.vsegdavkusno.ru/recipes/zharkoe-iz-svininy-s-ovoshchami-v-yablochnom-souse',
        'https://www.vsegdavkusno.ru/recipes/funchoza-s-ovoshchami-i-s-kuricey-po-koreyski',
        'https://www.vsegdavkusno.ru/recipes/pyure-iz-kartofelya-i-luka-poreya',
        'https://www.vsegdavkusno.ru/recipes/tushenaya-govyadina-v-souse-podlive',
        'https://www.vsegdavkusno.ru/recipes/albondigas-ispanskie-myasnye-frikadelki',
        'https://www.vsegdavkusno.ru/recipes/tvorozhno-ovoshchnye-keksy',
        'https://www.vsegdavkusno.ru/recipes/hrustyashchie-kurinye-krylyshki-v-duhovke',
        'https://www.vsegdavkusno.ru/recipes/gedlibzhe-kurica-v-smetane',
        'https://www.vsegdavkusno.ru/recipes/bigos-polskaya-kuhnya',
        'https://www.vsegdavkusno.ru/recipes/svinina-s-ovoshchami-po-kitayski',
        'https://www.vsegdavkusno.ru/recipes/briam-grecheskoe-ragu-iz-ovoshchey',
        'https://www.vsegdavkusno.ru/recipes/tefteli-v-belom-souse',
        'https://www.vsegdavkusno.ru/recipes/bulgur-s-ovoshchami',
        'https://www.vsegdavkusno.ru/recipes/kartofelnaya-zapekanka-s-kvashenoy-kapustoy-nemeckaya-kuhnya',
        'https://www.vsegdavkusno.ru/recipes/grechotto-s-gribami',
        'https://www.vsegdavkusno.ru/recipes/karri-ovoshchnoe-ragu',
        'https://www.vsegdavkusno.ru/recipes/brizol',
        'https://www.vsegdavkusno.ru/recipes/risovaya-kasha-na-moloke',
        'https://www.vsegdavkusno.ru/recipes/pyure-iz-zelenogo-goroshka',
        'https://www.vsegdavkusno.ru/recipes/file-treski-zapechennoe-v-duhovke',
        'https://www.vsegdavkusno.ru/recipes/tushenaya-kurica-s-ovoshchami-v-gorchichnom-souse',
        'https://www.vsegdavkusno.ru/recipes/zharenyij-kartofel-s-kapustoj-i-bekonom',
        'https://www.vsegdavkusno.ru/recipes/kuskus-so-svezhimi-ovoshhami',
        'https://www.vsegdavkusno.ru/recipes/goroshnicza',
        'https://www.vsegdavkusno.ru/recipes/zharenyie-krevetki-v-soevom-souse',
        'https://www.vsegdavkusno.ru/recipes/manchzhurskaya-kurica',
        'https://www.vsegdavkusno.ru/recipes/kroshka-kartoshka',
        'https://www.vsegdavkusno.ru/recipes/pasta-s-gribami.-italyanskaya-kuxnya',
        'https://www.vsegdavkusno.ru/recipes/rizotto-s-gribami',
        'https://www.vsegdavkusno.ru/recipes/perlovaya-kasha-s-ovoshhami',
        'https://www.vsegdavkusno.ru/recipes/makarony-po-flotski-s-tushenkoy',
        'https://www.vsegdavkusno.ru/recipes/ryiba-pod-marinadom',
        'https://www.vsegdavkusno.ru/recipes/kurinaya-pechen-po-bolgarski',
        'https://www.vsegdavkusno.ru/recipes/zharkoe-iz-kuriczyi-s-kartofelem',
        'https://www.vsegdavkusno.ru/recipes/kuricza-v-apelsinovom-souse',
        'https://www.vsegdavkusno.ru/recipes/flamandskoe-ragu-iz-govyadinyi',
        'https://www.vsegdavkusno.ru/recipes/kuricza-s-kabachkami',
        'https://www.vsegdavkusno.ru/recipes/funchoza-s-myasom',
        'https://www.vsegdavkusno.ru/recipes/solyanka-po-gruzinski',
        'https://www.vsegdavkusno.ru/recipes/legkij-franczuzskij-ratatuj',
        'https://www.vsegdavkusno.ru/recipes/izmir-kyofte',
        'https://www.vsegdavkusno.ru/recipes/grechnevaya-kasha-s-kabachkami',
        'https://www.vsegdavkusno.ru/recipes/kuricza-s-ovoshhami-v-duxovke',
        'https://www.vsegdavkusno.ru/recipes/kurinyie-kryilyishki-dvojnoj-obzharki',
        'https://www.vsegdavkusno.ru/recipes/vkusnyie-lenivyie-vareniki',
        'https://www.vsegdavkusno.ru/recipes/kuricza-po-oxotnichi',
        'https://www.vsegdavkusno.ru/recipes/sloppy-joe',
        'https://www.vsegdavkusno.ru/recipes/zharenyie-ovoshhi',
        'https://www.vsegdavkusno.ru/recipes/huevos-rotos',
        'https://www.vsegdavkusno.ru/recipes/makaronyi-s-tomatami-kabachkami-i-myasnyim-farshem',
        'https://www.vsegdavkusno.ru/recipes/kuricza-teriyaki-s-risom',
        'https://www.vsegdavkusno.ru/recipes/zharenaja-kartoshka-s-lisichkami',
        'https://www.vsegdavkusno.ru/recipes/skipperlabskovs',
        'https://www.vsegdavkusno.ru/recipes/rumyanaya-kuricza-s-risom',
        'https://www.vsegdavkusno.ru/recipes/farshirovannyij-perecz-v-aziatskom-stile',
        'https://www.vsegdavkusno.ru/recipes/pechen-po-stroganovski',
        'https://www.vsegdavkusno.ru/recipes/buglama',
        
    ]

    recipes = []

    for link in recipe_links:
        recipe_data = get_recipe_data(link)
        recipes.append(recipe_data)

    # Сохранение данных в JSON-файл
    with open('recipes.json', 'w', encoding='utf-8') as file:
        json.dump(recipes, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
