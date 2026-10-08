"""Build the original educational corpus; category is never a model input."""
import csv
from pathlib import Path

GROUPS = {
    'Space exploration': [
        'The spacecraft entered orbit around the planet. The mission collected images from a satellite and sent space observations to Earth.',
        'Astronauts aboard the space station repaired a satellite. Their mission studied how orbit affects equipment and human movement.',
        'A rocket launched the spacecraft toward the moon. Mission engineers adjusted the orbit and monitored the space journey.',
        'The telescope observed a distant planet and its moon. Space researchers used satellite images to study the orbit of the planet.',
        'Engineers tested a rocket engine for the next space mission. The spacecraft will carry a satellite into orbit around Earth.',
        'The moon mission landed a spacecraft near a crater. Astronauts collected rock samples while a satellite measured the lunar surface.',
        'A satellite tracked weather patterns from orbit. The space mission used telescope observations to compare Earth and another planet.',
        'The spacecraft changed orbit to inspect a planet. A rocket booster supported the mission while astronauts prepared space experiments.'
    ],
    'Sport and cricket': [
        'The cricket team won the match after the captain scored runs. Players praised the coach for planning the final tournament game.',
        'The coach trained players before the cricket tournament. The team practiced batting to score more runs in the next match.',
        'A cricket player hit a boundary in the final match. The captain helped the team defend its runs and win the tournament.',
        'The team selected young players for the cricket match. Their coach focused on batting technique and tournament preparation.',
        'The captain scored fifty runs in the cricket tournament. The team celebrated as players completed a difficult match.',
        'Rain delayed the cricket match but the players returned. The coach asked the team to protect wickets and build runs.',
        'The cricket tournament attracted supporters to the stadium. The captain and coach discussed team strategy before the match.',
        'Players improved their batting through regular practice. The cricket team scored enough runs to reach the tournament final.'
    ],
    'Healthcare': [
        'The hospital doctor examined a patient with fever. Medical treatment and medicine helped improve patient health after the diagnosis.',
        'A doctor discussed treatment with hospital nurses. The patient received medicine while medical tests checked heart health.',
        'Medical researchers tested a vaccine to prevent disease. The hospital doctor explained the treatment and monitored patient health.',
        'The patient visited the hospital for a health check. A doctor used medical tests to choose medicine and plan treatment.',
        'Hospital nurses tracked patient recovery after surgery. The doctor adjusted treatment and medicine to support better health.',
        'The doctor recommended exercise for heart health. Medical advice and regular treatment helped the patient avoid another hospital visit.',
        'A hospital opened a medical clinic for disease prevention. Each patient received health advice and medicine from a doctor.',
        'Medical staff reviewed patient records in the hospital. The doctor compared treatment options and checked whether medicine improved health.'
    ],
    'Computing and software': [
        'The software developer wrote code for a computer application. The program processed data and stored results on a network server.',
        'A computer program analysed data from the network. The developer updated software code to improve server performance.',
        'The developer tested application code before release. Software errors affected data processing on the computer server.',
        'The network server stored application data securely. A developer revised the software program and checked computer access.',
        'Students learned to write code for a computer program. Their software application retrieved data through a network server.',
        'The developer fixed a software bug in the application. New code reduced data loss when the computer contacted the server.',
        'A network failure interrupted the computer program. The developer restored the server and tested software data recovery.',
        'The software application used a database to organise data. The developer reviewed code and improved network server communication.'
    ]
}

if __name__ == '__main__':
    with Path(__file__).with_name('corpus.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'category', 'text'])
        for group, texts in GROUPS.items():
            for i, text in enumerate(texts, 1):
                writer.writerow([group.split()[0].lower()+f'_{i:02d}', group, text])
