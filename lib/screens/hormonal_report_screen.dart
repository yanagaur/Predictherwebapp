import 'package:flutter/material.dart';
import 'package:predicther/screens/detailedquestionnairescreen_no.dart';
import '../detailed_questionnaire_screen.dart';

class HormonalReportScreen extends StatelessWidget {
  const HormonalReportScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final List<String> facts = [
      'Regular monitoring of hormonal levels can help in early detection of various health issues.',
      'Hormones play a crucial role in regulating metabolism, growth, and mood.',
      'Imbalances in hormone levels can lead to conditions like diabetes, thyroid disorders, and infertility.',
      'Lifestyle factors such as diet, stress, and sleep can significantly impact hormonal balance.',
      'Consulting a healthcare provider for regular hormonal check-ups can contribute to overall well-being.',
    ];

    final PageController pageController = PageController();

    return Scaffold(
      appBar: AppBar(
        title: Text('Hormonal Report'),
        backgroundColor: Colors.pink,
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: <Widget>[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(16.0),
                    child: Card(
                      elevation: 0,
                      margin: EdgeInsets.symmetric(vertical: 80),
                      child: Padding(
                        padding: const EdgeInsets.all(30.0),
                        child: Text(
                          'Do you have your hormonal reports?',
                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                          textAlign: TextAlign.center,
                        ),
                      ),
                    ),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      ElevatedButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => DetailedQuestionnaireScreen(hasHormonalReports: true),
                            ),
                          );
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.pink,
                          foregroundColor: Colors.white,
                          padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8.0),
                          ),
                        ),
                        child: Text('Yes'),
                      ),
                      SizedBox(width: 10),
                      ElevatedButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => DetailedQuestionnaireScreen_no(hasHormonalReports: false),
                            ),
                          );
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.pink,
                          foregroundColor: Colors.white,
                          padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8.0),
                          ),
                        ),
                        child: Text('No'),
                      ),
                    ],
                  ),
                  SizedBox(height: 20),
                ],
              ),
            ),
          ),
          ClipRRect(
            borderRadius: BorderRadius.vertical(top: Radius.circular(16.0)),
            child: Card(
              color: Colors.white,
              elevation: 4,
              margin: EdgeInsets.all(16.0),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.info_outline, color: Colors.pink),
                        SizedBox(width: 10),
                        Text(
                          'Do you know?',
                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                    SizedBox(height: 10),
                    Container(
                      height: 150, // Adjust height as needed
                      child: PageView.builder(
                        controller: pageController,
                        itemCount: facts.length,
                        itemBuilder: (context, index) {
                          return Column(
                            children: [
                              Text(
                                facts[index],
                                style: TextStyle(fontSize: 16),
                                textAlign: TextAlign.left,
                              ),
                              SizedBox(height: 10),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                children: [
                                  if (index > 0)
                                    IconButton(
                                      icon: Icon(Icons.arrow_back),
                                      onPressed: () {
                                        pageController.previousPage(
                                          duration: Duration(milliseconds: 300),
                                          curve: Curves.easeInOut,
                                        );
                                      },
                                    ),
                                  if (index < facts.length - 1)
                                    IconButton(
                                      icon: Icon(Icons.arrow_forward),
                                      onPressed: () {
                                        pageController.nextPage(
                                          duration: Duration(milliseconds: 300),
                                          curve: Curves.easeInOut,
                                        );
                                      },
                                    ),
                                ],
                              ),
                            ],
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
